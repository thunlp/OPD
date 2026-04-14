# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import json
import os
import uuid
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from pprint import pprint
from typing import Optional

import numpy as np
import ray
import torch
from omegaconf import OmegaConf, open_dict
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from verl import DataProto
from verl.experimental.dataset.sampler import AbstractCurriculumSampler
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.config import AlgoConfig
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    process_validation_metrics,
)
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.trainer.ppo.utils import Role, WorkerType, need_critic, need_reference_policy, need_reward_model
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path, should_save_ckpt_esi
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.debug import marked_timer
from verl.utils.metric import reduce_metrics
from verl.utils.rollout_skip import RolloutSkip
from verl.utils.seqlen_balancing import calculate_workload, get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.torch_functional import masked_mean
from verl.utils.tracking import ValidationGenerationsLogger


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    """

    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        """Create Ray resource pools for distributed training.

        Initializes resource pools based on the resource pool specification,
        with each pool managing GPU resources across multiple nodes.
        For FSDP backend, uses max_colocate_count=1 to merge WorkerGroups.
        For Megatron backend, uses max_colocate_count>1 for different models.
        """
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1
            # that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(
                process_on_nodes=process_on_nodes, use_gpu=True, max_colocate_count=1, name_prefix=resource_pool_name
            )
            self.resource_pool_dict[resource_pool_name] = resource_pool

        self._check_resource_available()

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]

    def get_n_gpus(self) -> int:
        """Get the number of gpus in this cluster."""
        return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])

    def _check_resource_available(self):
        """Check if the resource pool can be satisfied in this ray cluster."""
        node_available_resources = ray._private.state.available_resources_per_node()
        node_available_gpus = {
            node: node_info.get("GPU", 0) if "GPU" in node_info else node_info.get("NPU", 0)
            for node, node_info in node_available_resources.items()
        }

        # check total required gpus can be satisfied
        total_available_gpus = sum(node_available_gpus.values())
        total_required_gpus = sum(
            [n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes]
        )
        if total_available_gpus < total_required_gpus:
            raise ValueError(
                f"Total available GPUs {total_available_gpus} is less than total desired GPUs {total_required_gpus}"
            )


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty="kl"):
    """Apply KL penalty to the token-level rewards.

    This function computes the KL divergence between the reference policy and current policy,
    then applies a penalty to the token-level rewards based on this divergence.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        kl_ctrl (core_algos.AdaptiveKLController): Controller for adaptive KL penalty.
        kl_penalty (str, optional): Type of KL penalty to apply. Defaults to "kl".

    Returns:
        tuple: A tuple containing:
            - The updated data with token-level rewards adjusted by KL penalty
            - A dictionary of metrics related to the KL penalty
    """
    response_mask = data.batch["response_mask"]
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]

    # compute kl between ref_policy and current policy
    # When apply_kl_penalty, algorithm.use_kl_in_reward=True, so the reference model has been enabled.
    kld = core_algos.kl_penalty(
        data.batch["old_log_probs"], data.batch["ref_log_prob"], kl_penalty=kl_penalty
    )  # (batch_size, response_length)
    kld = kld * response_mask
    beta = kl_ctrl.value

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch["token_level_rewards"] = token_level_rewards

    metrics = {"actor/reward_kl_penalty": current_kl, "actor/reward_kl_penalty_coeff": beta}

    return data, metrics


def compute_response_mask(data: DataProto):
    """Compute the attention mask for the response part of the sequence.

    This function extracts the portion of the attention mask that corresponds to the model's response,
    which is used for masking computations that should only apply to response tokens.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.

    Returns:
        torch.Tensor: The attention mask for the response tokens.
    """
    responses = data.batch["responses"]
    response_length = responses.size(1)
    attention_mask = data.batch["attention_mask"]
    return attention_mask[:, -response_length:]


def compute_advantage(
    data: DataProto,
    adv_estimator: AdvantageEstimator,
    gamma: float = 1.0,
    lam: float = 1.0,
    num_repeat: int = 1,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
) -> DataProto:
    """Compute advantage estimates for policy optimization.

    This function computes advantage estimates using various estimators like GAE, GRPO, REINFORCE++, etc.
    The advantage estimates are used to guide policy optimization in RL algorithms.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        adv_estimator (AdvantageEstimator): The advantage estimator to use (e.g., GAE, GRPO, REINFORCE++).
        gamma (float, optional): Discount factor for future rewards. Defaults to 1.0.
        lam (float, optional): Lambda parameter for GAE. Defaults to 1.0.
        num_repeat (int, optional): Number of times to repeat the computation. Defaults to 1.
        norm_adv_by_std_in_grpo (bool, optional): Whether to normalize advantages by standard deviation in
            GRPO. Defaults to True.
        config (dict, optional): Configuration dictionary for algorithm settings. Defaults to None.

    Returns:
        DataProto: The updated data with computed advantages and returns.
    """
    # Back-compatible with trainers that do not compute response mask in fit
    if "response_mask" not in data.batch.keys():
        data.batch["response_mask"] = compute_response_mask(data)
    # prepare response group
    if adv_estimator == AdvantageEstimator.GAE:
        # Compute advantages and returns using Generalized Advantage Estimation (GAE)
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=data.batch["token_level_rewards"],
            values=data.batch["values"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
            lam=lam,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        if config.get("use_pf_ppo", False):
            data = core_algos.compute_pf_ppo_reweight_data(
                data,
                config.pf_ppo.get("reweight_method"),
                config.pf_ppo.get("weight_pow"),
            )
    elif adv_estimator == AdvantageEstimator.GRPO:
        # Initialize the mask for GRPO calculation
        grpo_calculation_mask = data.batch["response_mask"]

        # Call compute_grpo_outcome_advantage with parameters matching its definition
        advantages, returns = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=grpo_calculation_mask,
            index=data.non_tensor_batch["uid"],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    else:
        # handle all other adv estimator type other than GAE and GRPO
        adv_estimator_fn = core_algos.get_adv_estimator_fn(adv_estimator)
        adv_kwargs = {
            "token_level_rewards": data.batch["token_level_rewards"],
            "response_mask": data.batch["response_mask"],
            "config": config,
        }
        if "uid" in data.non_tensor_batch:  # optional
            adv_kwargs["index"] = data.non_tensor_batch["uid"]
        if "true_reward_score" in data.batch: # optional
            adv_kwargs["true_reward_score"] = data.batch["true_reward_score"]
        if "reward_baselines" in data.batch:  # optional
            adv_kwargs["reward_baselines"] = data.batch["reward_baselines"]

        # calculate advantage estimator
        res = adv_estimator_fn(**adv_kwargs)
        if len(res) == 2:
            advantages, returns = res
        elif len(res) == 3:
            advantages, returns, extra_metrics = res
            for k, v in extra_metrics.items():
                data.batch[k] = v
        else:
            raise ValueError("Invalid return from adv_estimator_fn")

        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    return data


class RayPPOTrainer:
    """Distributed PPO trainer using Ray for scalable reinforcement learning.

    This trainer orchestrates distributed PPO training across multiple nodes and GPUs,
    managing actor rollouts, critic training, and reward computation with Ray backend.
    Supports various model architectures including FSDP, Megatron, vLLM, and SGLang integration.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: type[RayWorkerGroup] = RayWorkerGroup,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
        device_name=None,
    ):
        """
        Initialize distributed PPO trainer with Ray backend.
        Note that this trainer runs on the driver process on a single CPU/GPU node.

        Args:
            config: Configuration object containing training parameters.
            tokenizer: Tokenizer used for encoding and decoding text.
            role_worker_mapping (dict[Role, WorkerType]): Mapping from roles to worker classes.
            resource_pool_manager (ResourcePoolManager): Manager for Ray resource pools.
            ray_worker_group_cls (RayWorkerGroup, optional): Class for Ray worker groups. Defaults to RayWorkerGroup.
            processor: Optional data processor, used for multimodal data
            reward_fn: Function for computing rewards during training.
            val_reward_fn: Function for computing rewards during validation.
            train_dataset (Optional[Dataset], optional): Training dataset. Defaults to None.
            val_dataset (Optional[Dataset], optional): Validation dataset. Defaults to None.
            collate_fn: Function to collate data samples into batches.
            train_sampler (Optional[Sampler], optional): Sampler for the training dataset. Defaults to None.
            device_name (str, optional): Device name for training (e.g., "cuda", "cpu"). Defaults to None.
        """

        # Store the tokenizer for text processing
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, "Currently, only support hybrid engine"

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f"{role_worker_mapping.keys()=}"

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = need_reference_policy(self.role_worker_mapping)
        self.use_rm = need_reward_model(self.role_worker_mapping)
        self.use_critic = need_critic(self.config)
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name if device_name else self.config.trainer.device
        self.validation_generations_logger = ValidationGenerationsLogger(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
        )

        # if ref_in_actor is True, the reference policy will be actor without lora applied
        self.ref_in_actor = (
            config.actor_rollout_ref.model.get("lora_rank", 0) > 0
            or config.actor_rollout_ref.model.get("lora_adapter_path") is not None
        )

        # define in-reward KL control
        # kl loss control currently not suppoorted
        if self.config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(self.config.algorithm.kl_ctrl)

        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)

    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler: Optional[Sampler]):
        """
        Creates the train and validation dataloaders.
        """
        # TODO: we have to make sure the batch size is divisible by the dp size
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler

        if train_dataset is None:
            train_dataset = create_rl_dataset(
                self.config.data.train_files,
                self.config.data,
                self.tokenizer,
                self.processor,
                max_samples=self.config.data.get("train_max_samples", -1),
            )
        if val_dataset is None:
            val_dataset = create_rl_dataset(
                self.config.data.val_files,
                self.config.data,
                self.tokenizer,
                self.processor,
                max_samples=self.config.data.get("val_max_samples", -1),
            )
        self.train_dataset, self.val_dataset = train_dataset, val_dataset

        if train_sampler is None:
            train_sampler = create_rl_sampler(self.config.data, self.train_dataset)
        if collate_fn is None:
            from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn

            collate_fn = default_collate_fn

        num_workers = self.config.data["dataloader_num_workers"]

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.get("gen_batch_size", self.config.data.train_batch_size),
            num_workers=num_workers,
            drop_last=True,
            collate_fn=collate_fn,
            sampler=train_sampler,
        )

        val_batch_size = self.config.data.val_batch_size  # Prefer config value if set
        if val_batch_size is None:
            val_batch_size = len(self.val_dataset)

        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=val_batch_size,
            num_workers=num_workers,
            shuffle=self.config.data.get("validation_shuffle", True),
            drop_last=False,
            collate_fn=collate_fn,
        )

        assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"
        assert len(self.val_dataloader) >= 1, "Validation dataloader is empty!"

        print(
            f"Size of train dataloader: {len(self.train_dataloader)}, Size of val dataloader: "
            f"{len(self.val_dataloader)}"
        )

        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")

        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = total_training_steps
        except Exception as e:
            print(f"Warning: Could not set total_training_steps in config. Structure missing? Error: {e}")

    def _dump_generations(self, inputs, outputs, gts, scores, reward_extra_infos_dict, dump_path):
        """Dump rollout/validation samples as JSONL."""
        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.join(dump_path, f"{self.global_steps}.jsonl")

        n = len(inputs)
        base_data = {
            "input": inputs,
            "output": outputs,
            "gts": gts,
            "score": scores,
            "step": [self.global_steps] * n,
        }

        for k, v in reward_extra_infos_dict.items():
            if len(v) == n:
                base_data[k] = v

        lines = []
        for i in range(n):
            entry = {k: v[i] for k, v in base_data.items()}
            lines.append(json.dumps(entry, ensure_ascii=False))

        with open(filename, "w") as f:
            f.write("\n".join(lines) + "\n")

        print(f"Dumped generations to {filename}")

    def _log_rollout_data(
        self, batch: DataProto, reward_extra_infos_dict: dict, timing_raw: dict, rollout_data_dir: str
    ):
        """Log rollout data to disk.
        Args:
            batch (DataProto): The batch containing rollout data
            reward_extra_infos_dict (dict): Additional reward information to log
            timing_raw (dict): Timing information for profiling
            rollout_data_dir (str): Directory path to save the rollout data
        """
        with marked_timer("dump_rollout_generations", timing_raw, color="green"):
            inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
            outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
            sample_gts = [item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in batch]

            reward_extra_infos_to_dump = reward_extra_infos_dict.copy()
            if "request_id" in batch.non_tensor_batch:
                reward_extra_infos_dict.setdefault(
                    "request_id",
                    batch.non_tensor_batch["request_id"].tolist(),
                )

            self._dump_generations(
                inputs=inputs,
                outputs=outputs,
                gts=sample_gts,
                scores=scores,
                reward_extra_infos_dict=reward_extra_infos_to_dump,
                dump_path=rollout_data_dir,
            )

    def _maybe_log_val_generations(self, inputs, outputs, scores):
        """Log a table of validation samples to the configured logger (wandb or swanlab)"""

        generations_to_log = self.config.trainer.log_val_generations

        if generations_to_log == 0:
            return

        import numpy as np

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, scores, strict=True))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        # Take first N samples after shuffling
        samples = samples[:generations_to_log]

        # Log to each configured logger
        self.validation_generations_logger.log(self.config.trainer.logger, samples, self.global_steps)

    def _get_gen_batch(self, batch: DataProto) -> DataProto:
        reward_model_keys = set({"data_source", "reward_model", "extra_info", "uid"}) & batch.non_tensor_batch.keys()

        # pop those keys for generation
        batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
        non_tensor_batch_keys_to_pop = set(batch.non_tensor_batch.keys()) - reward_model_keys
        gen_batch = batch.pop(
            batch_keys=batch_keys_to_pop,
            non_tensor_batch_keys=list(non_tensor_batch_keys_to_pop),
        )

        # For agent loop, we need reward model keys to compute score.
        if self.async_rollout_mode:
            gen_batch.non_tensor_batch.update(batch.non_tensor_batch)

        return gen_batch

    def _validate(self):
        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_gts = []
        sample_scores = []
        sample_turns = []
        sample_uids = []

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)

            if "uid" not in test_batch.non_tensor_batch:
                test_batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(test_batch.batch))], dtype=object
                )

            # repeat test batch
            test_batch = test_batch.repeat(
                repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True
            )

            # we only do validation on rule-based rm
            if self.config.reward_model.enable and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model":
                return {}

            # Store original inputs
            input_ids = test_batch.batch["input_ids"]
            # TODO: Can we keep special tokens except for padding tokens?
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)
            sample_uids.extend(test_batch.non_tensor_batch["uid"])

            ground_truths = [
                item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in test_batch
            ]
            sample_gts.extend(ground_truths)

            test_gen_batch = self._get_gen_batch(test_batch)
            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
                "global_steps": self.global_steps,
            }
            print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

            # pad to be divisible by dp_size
            size_divisor = (
                self.actor_rollout_wg.world_size
                if not self.async_rollout_mode
                else self.config.actor_rollout_ref.rollout.agent.num_workers
            )
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, size_divisor)
            if not self.async_rollout_mode:
                test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
            else:
                test_output_gen_batch_padded = self.async_rollout_manager.generate_sequences(test_gen_batch_padded)

            # unpad
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)

            print("validation generation end")

            # Store generated outputs
            output_ids = test_output_gen_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)

            test_batch = test_batch.union(test_output_gen_batch)
            test_batch.meta_info["validate"] = True

            # evaluate using reward_function
            if self.val_reward_fn is None:
                raise ValueError("val_reward_fn must be provided for validation.")
            result = self.val_reward_fn(test_batch, return_dict=True)
            reward_tensor = result["reward_tensor"]
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_extra_infos_dict["reward"].extend(scores)
            if "reward_extra_info" in result:
                for key, lst in result["reward_extra_info"].items():
                    reward_extra_infos_dict[key].extend(lst)

            # collect num_turns of each prompt
            if "__num_turns__" in test_batch.non_tensor_batch:
                sample_turns.append(test_batch.non_tensor_batch["__num_turns__"])

            data_source_lst.append(test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0]))

        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        # dump generations
        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        if val_data_dir:
            self._dump_generations(
                inputs=sample_inputs,
                outputs=sample_outputs,
                gts=sample_gts,
                scores=sample_scores,
                reward_extra_infos_dict=reward_extra_infos_dict,
                dump_path=val_data_dir,
            )

        for key_info, lst in reward_extra_infos_dict.items():
            assert len(lst) == 0 or len(lst) == len(sample_scores), f"{key_info}: {len(lst)=}, {len(sample_scores)=}"

        data_sources = np.concatenate(data_source_lst, axis=0)

        data_src2var2metric2val = process_validation_metrics(data_sources, sample_uids, reward_extra_infos_dict)
        metric_dict = {}
        for data_source, var2metric2val in data_src2var2metric2val.items():
            core_var = "acc" if "acc" in var2metric2val else "reward"
            for var_name, metric2val in var2metric2val.items():
                n_max = max([int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys()])
                for metric_name, metric_val in metric2val.items():
                    if (
                        (var_name == core_var)
                        and any(metric_name.startswith(pfx) for pfx in ["mean", "maj", "best"])
                        and (f"@{n_max}" in metric_name)
                    ):
                        metric_sec = "val-core"
                    else:
                        metric_sec = "val-aux"
                    pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}"
                    metric_dict[pfx] = metric_val

        if len(sample_turns) > 0:
            sample_turns = np.concatenate(sample_turns)
            metric_dict["val-aux/num_turns/min"] = sample_turns.min()
            metric_dict["val-aux/num_turns/max"] = sample_turns.max()
            metric_dict["val-aux/num_turns/mean"] = sample_turns.mean()

        return metric_dict

    def init_workers(self):
        """Initialize distributed training workers using Ray backend.

        Creates:
        1. Ray resource pools from configuration
        2. Worker groups for each role (actor, critic, etc.)
        """
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRollout],
                config=self.config.actor_rollout_ref,
                role=str(Role.ActorRollout),
            )
            self.resource_pool_to_cls[resource_pool][str(Role.ActorRollout)] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cfg = omega_conf_to_dataclass(self.config.critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=critic_cfg)
            self.resource_pool_to_cls[resource_pool][str(Role.Critic)] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RefPolicy],
                config=self.config.actor_rollout_ref,
                role=str(Role.RefPolicy),
            )
            self.resource_pool_to_cls[resource_pool][str(Role.RefPolicy)] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool][str(Role.RewardModel)] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`.
        # Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout
        if OmegaConf.select(self.config.global_profiler, "steps") is not None:
            wg_kwargs["profile_steps"] = OmegaConf.select(self.config.global_profiler, "steps")
            # Only require nsight worker options when tool is nsys
            if OmegaConf.select(self.config.global_profiler, "tool") == "nsys":
                assert (
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                    is not None
                ), "worker_nsight_options must be set when using nsys with profile_steps"
                wg_kwargs["worker_nsight_options"] = OmegaConf.to_container(
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                )
        wg_kwargs["device_name"] = self.device_name

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(
                resource_pool=resource_pool,
                ray_cls_with_init=worker_dict_cls,
                **wg_kwargs,
            )
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)

        if self.use_critic:
            self.critic_wg = all_wg[str(Role.Critic)]
            self.critic_wg.init_model()

        if self.use_reference_policy and not self.ref_in_actor:
            self.ref_policy_wg = all_wg[str(Role.RefPolicy)]
            self.ref_policy_wg.init_model()

        self.rm_wg = None
        # initalization of rm_wg will be deprecated in the future
        if self.use_rm:
            self.rm_wg = all_wg[str(Role.RewardModel)]
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg[str(Role.ActorRollout)]
        self.actor_rollout_wg.init_model()

        # create async rollout manager and request scheduler
        self.async_rollout_mode = False
        if self.config.actor_rollout_ref.rollout.mode == "async":
            from verl.experimental.agent_loop import AgentLoopManager

            self.async_rollout_mode = True
            self.async_rollout_manager = AgentLoopManager(
                config=self.config, worker_group=self.actor_rollout_wg, rm_wg=self.rm_wg
            )

    def _save_checkpoint(self):
        from verl.utils.fs import local_mkdir_safe

        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(
            self.config.trainer.default_local_dir, f"global_step_{self.global_steps}"
        )

        print(f"local_global_step_folder: {local_global_step_folder}")
        actor_local_path = os.path.join(local_global_step_folder, "actor")

        actor_remote_path = (
            None
            if self.config.trainer.default_hdfs_dir is None
            else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "actor")
        )

        remove_previous_ckpt_in_save = self.config.trainer.get("remove_previous_ckpt_in_save", False)
        if remove_previous_ckpt_in_save:
            print(
                "Warning: remove_previous_ckpt_in_save is deprecated,"
                + " set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead"
            )
        max_actor_ckpt_to_keep = (
            self.config.trainer.get("max_actor_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )
        max_critic_ckpt_to_keep = (
            self.config.trainer.get("max_critic_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )

        self.actor_rollout_wg.save_checkpoint(
            actor_local_path, actor_remote_path, self.global_steps, max_ckpt_to_keep=max_actor_ckpt_to_keep
        )

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, str(Role.Critic))
            critic_remote_path = (
                None
                if self.config.trainer.default_hdfs_dir is None
                else os.path.join(
                    self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", str(Role.Critic)
                )
            )
            self.critic_wg.save_checkpoint(
                critic_local_path, critic_remote_path, self.global_steps, max_ckpt_to_keep=max_critic_ckpt_to_keep
            )

        # save dataloader
        local_mkdir_safe(local_global_step_folder)
        dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)

        # latest checkpointed iteration tracker (for atomic usage)
        local_latest_checkpointed_iteration = os.path.join(
            self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt"
        )
        with open(local_latest_checkpointed_iteration, "w") as f:
            f.write(str(self.global_steps))

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == "disable":
            # NOTE: while there is no checkpoint to load, we still need to offload the model and optimizer to CPU
            self.actor_rollout_wg.load_checkpoint(None)
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError("load from hdfs is not implemented yet")
        else:
            checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                print("Training from scratch")
                self.actor_rollout_wg.load_checkpoint(None)
                return 0
        else:
            if self.config.trainer.resume_mode == "resume_path":
                assert isinstance(self.config.trainer.resume_from_path, str), "resume ckpt must be str type"
                assert "global_step_" in self.config.trainer.resume_from_path, (
                    "resume ckpt must specify the global_steps"
                )
                global_step_folder = self.config.trainer.resume_from_path
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f"Load from checkpoint folder: {global_step_folder}")
        # set global step
        self.global_steps = int(global_step_folder.split("global_step_")[-1])

        print(f"Setting global step to {self.global_steps}")
        print(f"Resuming from {global_step_folder}")

        actor_path = os.path.join(global_step_folder, "actor")
        critic_path = os.path.join(global_step_folder, str(Role.Critic))
        # load actor
        self.actor_rollout_wg.load_checkpoint(
            actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
        )
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(
                critic_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
            )

        # load dataloader,
        # TODO: from remote not implemented yet
        dataloader_local_path = os.path.join(global_step_folder, "data.pt")
        if os.path.exists(dataloader_local_path):
            dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")

    def _start_profiling(self, do_profile: bool) -> None:
        """Start profiling for all worker groups if profiling is enabled."""
        if do_profile:
            self.actor_rollout_wg.start_profile(role="e2e", profile_step=self.global_steps)
            if self.use_reference_policy:
                self.ref_policy_wg.start_profile(profile_step=self.global_steps)
            if self.use_critic:
                self.critic_wg.start_profile(profile_step=self.global_steps)
            if self.use_rm:
                self.rm_wg.start_profile(profile_step=self.global_steps)

    def _stop_profiling(self, do_profile: bool) -> None:
        """Stop profiling for all worker groups if profiling is enabled."""
        if do_profile:
            self.actor_rollout_wg.stop_profile()
            if self.use_reference_policy:
                self.ref_policy_wg.stop_profile()
            if self.use_critic:
                self.critic_wg.stop_profile()
            if self.use_rm:
                self.rm_wg.stop_profile()

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix="global_seqlen", keep_minibatch=False):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1)  # (train_batch_size,)
        global_seqlen_lst = calculate_workload(global_seqlen_lst)
        world_size = self.actor_rollout_wg.world_size
        if keep_minibatch:
            # Decouple the DP balancing and mini-batching.
            minibatch_size = self.config.actor_rollout_ref.actor.get("ppo_mini_batch_size")
            minibatch_num = len(global_seqlen_lst) // minibatch_size
            global_partition_lst = [[] for _ in range(world_size)]
            for i in range(minibatch_num):
                rearrange_minibatch_lst = get_seqlen_balanced_partitions(
                    global_seqlen_lst[i * minibatch_size : (i + 1) * minibatch_size],
                    k_partitions=world_size,
                    equal_size=True,
                )
                for j, part in enumerate(rearrange_minibatch_lst):
                    global_partition_lst[j].extend([x + minibatch_size * i for x in part])
        else:
            global_partition_lst = get_seqlen_balanced_partitions(
                global_seqlen_lst, k_partitions=world_size, equal_size=True
            )
        # Place smaller micro-batches at both ends to reduce the bubbles in pipeline parallel.
        for idx, partition in enumerate(global_partition_lst):
            partition.sort(key=lambda x: (global_seqlen_lst[x], x))
            ordered_partition = partition[::2] + partition[1::2][::-1]
            global_partition_lst[idx] = ordered_partition
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst, partitions=global_partition_lst, prefix=logging_prefix
        )
        metrics.update(global_balance_stats)

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        if self.config.actor_rollout_ref.rollout.get("skip_rollout", False):
            rollout_skip = RolloutSkip(self.config, self.actor_rollout_wg)
            rollout_skip.wrap_generate_sequences()

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None
        self.max_steps_duration = 0

        prev_step_profile = False
        curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        next_step_profile = False

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}

                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(
                        not prev_step_profile and curr_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                batch: DataProto = DataProto.from_single_dict(batch_dict)

                # add uid to batch
                batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                )

                gen_batch = self._get_gen_batch(batch)

                # pass global_steps to trace
                gen_batch.meta_info["global_steps"] = self.global_steps
                gen_batch_output = gen_batch.repeat(
                    repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True
                )

                is_last_step = self.global_steps >= self.total_training_steps
                with marked_timer("step", timing_raw):
                    # generate a batch
                    with marked_timer("gen", timing_raw, color="red"):
                        if not self.async_rollout_mode:
                            gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch_output)
                        else:
                            gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch_output)

                        timing_raw.update(gen_batch_output.meta_info["timing"])
                        gen_batch_output.meta_info.pop("timing", None)

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        if self.reward_fn is None:
                            raise ValueError("A reward_fn is required for REMAX advantage estimation.")

                        with marked_timer("gen_max", timing_raw, color="purple"):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            if not self.async_rollout_mode:
                                gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)
                            else:
                                gen_baseline_output = self.async_rollout_manager.generate_sequences(gen_baseline_batch)
                            batch = batch.union(gen_baseline_output)
                            # compute reward model score on batch
                            rm_scores = None
                            if self.use_rm and "rm_scores" not in batch.batch.keys():
                                # pass global_steps and is_plot config to rm_wg
                                batch.meta_info["global_steps"] = self.global_steps
                                batch.meta_info["is_plot"] = self.config.trainer.get("is_plot", False)
                                rm_scores = self.rm_wg.compute_rm_score(batch)
                                batch = batch.union(rm_scores)
                            reward_baseline_tensor, _ = compute_reward(batch, self.reward_fn)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            keys_to_pop = set(gen_baseline_output.batch.keys())
                            if rm_scores is not None:
                                keys_to_pop.update(rm_scores.batch.keys())
                            batch.pop(batch_keys=list(keys_to_pop))

                            batch.batch["reward_baselines"] = reward_baseline_tensor

                            del rm_scores, gen_baseline_batch, gen_baseline_output
                    # repeat to align with repeated responses in rollout
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    batch = batch.union(gen_batch_output)

                    if "response_mask" not in batch.batch.keys():
                        batch.batch["response_mask"] = compute_response_mask(batch)
                    # Balance the number of valid tokens across DP ranks.
                    # NOTE: This usually changes the order of data in the `batch`,
                    # which won't affect the advantage calculation (since it's based on uid),
                    # but might affect the loss calculation (due to the change of mini-batching).
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    with marked_timer("reward", timing_raw, color="yellow"):
                        # compute reward model score
                        if self.use_rm and "rm_scores" not in batch.batch.keys():
                            with marked_timer("compute_log_prob", timing_raw, color="blue"):
                                # First forward, get student top k ids and log probs
                                print("First forward, get student top k ids and log probs")
                                old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)

                                # if "entropys" in old_log_prob.batch.keys():
                                #    old_log_prob.batch.pop("entropys")
                                batch = batch.union(old_log_prob)

                            # Get Top-K parameters from config
                            top_k = self.config.actor_rollout_ref.rollout.get("log_prob_top_k", 0)
                            strategy = self.config.actor_rollout_ref.rollout.get("top_k_strategy", "only_stu")
                            kl_estimator = self.config.actor_rollout_ref.rollout.get("kl_estimator", "k1")
                            reward_weight_mode = self.config.actor_rollout_ref.rollout.get("reward_weight_mode", "student_p")

                            # pass global_steps and is_plot config to rm_wg
                            batch.meta_info["global_steps"] = self.global_steps
                            batch.meta_info["is_plot"] = self.config.trainer.get("is_plot", False)
                            teacher_temperature = self.config.actor_rollout_ref.rollout.get("teacher_temperature", 1.0)

                            batch.meta_info["log_prob_top_k"] = top_k
                            batch.meta_info["top_k_strategy"] = strategy
                            batch.meta_info["kl_estimator"] = kl_estimator
                            batch.meta_info["reward_weight_mode"] = reward_weight_mode
                            batch.meta_info["teacher_temperature"] = teacher_temperature
                            
                            with marked_timer("compute_rm_score", timing_raw, color="magenta"):
                                teacher_data = self.rm_wg.compute_rm_score(batch)
                                batch = batch.union(teacher_data)

                            if top_k > 0:
                                # All distillation reward calculation is now moved to GPU worker (actor_rollout_wg)
                                # for efficiency and to reduce CPU tensor ops.
                                # compute_distillation_reward computes S_on_T and then rm_scores.
                                with marked_timer("compute_distillation_reward", timing_raw, color="orange"):
                                    distillation_output = self.actor_rollout_wg.compute_distillation_reward(batch)
                                    batch = batch.union(distillation_output)
                        
                        # Plot overlapping tokens for Reverse KL
                        if (self.global_steps == 1 or self.global_steps % 10 == 0) and "student_valid_counts" in batch.batch.keys():
                            try:
                                import matplotlib.pyplot as plt
                                import swanlab

                                response_mask = batch.batch["response_mask"]
                                valid_denom = response_mask.sum(dim=0) + 1e-6

                                plot_data = {}
                                
                                # Calculate Student Candidates
                                if "student_valid_counts" in batch.batch.keys():
                                    student_counts = batch.batch["student_valid_counts"].float()
                                    avg_student_counts = (student_counts * response_mask).sum(dim=0) / valid_denom
                                    plot_data["Student"] = avg_student_counts.detach().cpu().numpy()
                                
                                # Calculate Teacher and Overlap Candidates if available
                                if "teacher_valid_counts" in batch.batch.keys():
                                    teacher_counts = batch.batch["teacher_valid_counts"].float()
                                    avg_teacher_counts = (teacher_counts * response_mask).sum(dim=0) / valid_denom
                                    plot_data["Teacher"] = avg_teacher_counts.detach().cpu().numpy()
                                    
                                if "overlap_mask" in batch.batch.keys():
                                    # overlap_mask is (BS, SeqLen, K), sum over K to get counts
                                    overlap_mask = batch.batch["overlap_mask"].float()
                                    overlap_counts = overlap_mask.sum(dim=-1)  # (BS, SeqLen)
                                    avg_overlap_counts = (overlap_counts * response_mask).sum(dim=0) / valid_denom
                                    plot_data["Overlap"] = avg_overlap_counts.detach().cpu().numpy()
                                
                                # Plot 1: Candidate Counts
                                plt.figure(figsize=(10, 6))
                                for label, data in plot_data.items():
                                    mean_val = data.mean()
                                    plt.plot(data, label=f"Avg {label} (mean: {mean_val:.2f})")
                                
                                plt.title(f"Avg Candidate Tokens per Position (Step {self.global_steps})")
                                plt.xlabel("Position")
                                plt.ylabel("Avg Candidate Count")
                                plt.legend()
                                plt.grid(True)
                                plt.tight_layout()
                                
                                count_plot = swanlab.Image(plt, caption=f"Candidate Counts (Step {self.global_steps})")
                                plt.close()
                                
                                # Plot 2: Ratios
                                log_payload = {"viz/candidate_counts": count_plot}
                                
                                if "Overlap" in plot_data and "Student" in plot_data and "Teacher" in plot_data:
                                    ratio_student = plot_data["Overlap"] / (plot_data["Student"] + 1e-6)
                                    ratio_teacher = plot_data["Overlap"] / (plot_data["Teacher"] + 1e-6)
                                    
                                    # Plot 2a: Overlap / Student
                                    plt.figure(figsize=(10, 6))
                                    plt.plot(ratio_student, label=f"Overlap / Student (mean: {ratio_student.mean():.2f})", color='tab:blue')
                                    plt.title(f"Overlap / Student Ratio (Step {self.global_steps})")
                                    plt.xlabel("Position")
                                    plt.ylabel("Ratio")
                                    plt.ylim(-0.05, 1.05)
                                    plt.legend()
                                    plt.grid(True)
                                    plt.tight_layout()
                                    
                                    ratio_student_plot = swanlab.Image(plt, caption=f"Overlap / Student Ratio (Step {self.global_steps})")
                                    plt.close()
                                    log_payload["viz/overlap_ratio_student"] = ratio_student_plot

                                    # Plot 2b: Overlap / Teacher
                                    plt.figure(figsize=(10, 6))
                                    plt.plot(ratio_teacher, label=f"Overlap / Teacher (mean: {ratio_teacher.mean():.2f})", color='tab:orange')
                                    plt.title(f"Overlap / Teacher Ratio (Step {self.global_steps})")
                                    plt.xlabel("Position")
                                    plt.ylabel("Ratio")
                                    plt.ylim(-0.05, 1.05)
                                    plt.legend()
                                    plt.grid(True)
                                    plt.tight_layout()
                                    
                                    ratio_teacher_plot = swanlab.Image(plt, caption=f"Overlap / Teacher Ratio (Step {self.global_steps})")
                                    plt.close()
                                    log_payload["viz/overlap_ratio_teacher"] = ratio_teacher_plot

                                logger.log(log_payload, step=self.global_steps)
                                print(f"Logged candidate plots to SwanLab at step {self.global_steps}")
                                
                            except Exception as e:
                                print(f"Error plotting candidate counts: {e}")
                        
                        
                        # Keep student_top_k_log_probs for potential use in policy loss computation
                        # Only pop temporary visualization data
                        if "student_valid_counts" in batch.batch.keys():
                             batch.batch.pop("student_valid_counts")
                        if "teacher_valid_counts" in batch.batch.keys():
                             batch.batch.pop("teacher_valid_counts")
                        if "overlap_counts" in batch.batch.keys():
                             batch.batch.pop("overlap_counts")


                        if self.config.reward_model.launch_reward_fn_async:
                            future_reward = compute_reward_async.remote(
                                data=batch, config=self.config, tokenizer=self.tokenizer
                            )
                        else:
                            reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)
                            if "format_mask" in reward_extra_infos_dict.keys():
                                batch.batch["format_mask"] = reward_extra_infos_dict["format_mask"]
                    
                    from verl.trainer.ppo.rollout_corr_helper import (
                        compute_rollout_correction_and_add_to_batch,
                        maybe_apply_rollout_correction,
                    )

                    rollout_corr_config = self.config.algorithm.get("rollout_correction", None)
                    need_recomputation = maybe_apply_rollout_correction(
                        batch=batch,
                        rollout_corr_config=rollout_corr_config,
                        policy_loss_config=self.config.actor_rollout_ref.actor.policy_loss,
                    )
                    if need_recomputation:
                        # Optimization: Reuse data if available from Distillation Phase
                        entropys = None
                        if "old_log_probs" in batch.batch.keys() and "entropys" in batch.batch.keys():
                             entropys = batch.batch["entropys"]
                             print("We don't need to re-merge old_log_probs, it's already there.")

                        else:
                             # Legacy Path: Must recompute if not present
                             with marked_timer("old_log_prob", timing_raw, color="blue"):
                                 old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                                 entropys = old_log_prob.batch["entropys"]
                                 batch = batch.union(old_log_prob)
                                 
                                 # Remove top-k keys from old_log_prob if they already exist in batch
                                 # (they may have been modified for union strategy)
                                 for key in ["student_top_k_ids", "student_top_k_log_probs"]:
                                     if key in batch.batch.keys() and key in old_log_prob.batch.keys():
                                         pass # Already handled by union? Warning: Union might overwrite if not careful.
                                         # The original code had a manual check here, but batch.union generally overwrites.
                                         # Assuming Actor's new log prob output is the "source of truth" if we recompute.

                        if entropys is not None:
                            response_masks = batch.batch["response_mask"]
                            if "format_mask" in batch.batch.keys():
                                response_masks = response_masks * batch.batch["format_mask"].unsqueeze(-1)
                            
                            loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                            entropy_agg = agg_loss(
                                loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode
                            )
                            metrics.update({"actor/entropy": entropy_agg.detach().item()})

                            # Compute teacher entropy metric if available
                            if "teacher_entropy" in batch.batch.keys():
                                teacher_entropy = batch.batch["teacher_entropy"]
                                teacher_entropy_agg = agg_loss(
                                    loss_mat=teacher_entropy, loss_mask=response_masks, loss_agg_mode=loss_agg_mode
                                )
                                metrics.update({"teacher/entropy": teacher_entropy_agg.detach().item()})

                            # Cleanup: We are done with entropys
                            if "entropys" in batch.batch.keys():
                                batch.batch.pop("entropys")


                            if "rollout_log_probs" in batch.batch.keys():
                                # TODO: we may want to add diff of probs too.
                                from verl.utils.debug.metrics import calculate_debug_metrics

                                metrics.update(calculate_debug_metrics(batch))

                    assert "old_log_probs" in batch.batch, f'"old_log_prob" not in {batch.batch.keys()=}'

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with marked_timer(str(Role.RefPolicy), timing_raw, color="olive"):
                            if not self.ref_in_actor:
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            else:
                                ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with marked_timer("values", timing_raw, color="cyan"):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with marked_timer("adv", timing_raw, color="brown"):
                        # we combine with rule-based rm
                        reward_extra_infos_dict: dict[str, list]
                        if self.config.reward_model.launch_reward_fn_async:
                            reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                        batch.batch["token_level_scores"] = reward_tensor

                        if "true_reward_score" in reward_extra_infos_dict:
                            true_reward_val = reward_extra_infos_dict["true_reward_score"]
                            if isinstance(true_reward_val, torch.Tensor):
                                batch.batch["true_reward_score"] = true_reward_val
                            else:
                                batch.batch["true_reward_score"] = torch.as_tensor(
                                    true_reward_val,
                                    device=reward_tensor.device,
                                    dtype=reward_tensor.dtype,
                                )
                        else:
                            batch.batch["true_reward_score"] = reward_tensor

                        if reward_extra_infos_dict:
                            batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(
                                batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                            )
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        # Compute rollout correction weights centrally (once per batch)
                        # This corrects for off-policy issues (policy mismatch, model staleness, etc.)
                        # Also computes off-policy diagnostic metrics (KL, PPL, etc.)
                        if rollout_corr_config is not None and "rollout_log_probs" in batch.batch:
                            batch, is_metrics = compute_rollout_correction_and_add_to_batch(batch, rollout_corr_config)
                            # IS and off-policy metrics already have rollout_corr/ prefix
                            metrics.update(is_metrics)

                        # compute advantages, executed on the driver process
                        norm_adv_by_std_in_grpo = self.config.algorithm.get(
                            "norm_adv_by_std_in_grpo", True
                        )  # GRPO adv normalization factor

                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            config=self.config.algorithm,
                        )
 

                        # --- Top-K Metrics Analysis (Chunked) ---
                        if "overlap_mask" in batch.batch.keys() and "advantages" in batch.batch.keys():
                            try:
                                overlap_mask = batch.batch["overlap_mask"].float() # (BS, SeqLen, K)
                                advantages = batch.batch["advantages"] # (BS, SeqLen, K) or (BS, SeqLen, 2K) for union
                                
                                response_mask = batch.batch["response_mask"] # (BS, SeqLen)
                                max_len = response_mask.shape[-1]
                                top_k = batch.meta_info.get("log_prob_top_k", 0)
                                strategy = batch.meta_info.get("top_k_strategy", "only_stu")
                                
                                # For union strategy, get teacher_in_student mask
                                teacher_in_student_mask = batch.batch.get("teacher_in_student_mask", None) # (BS, SeqLen, K) or None
                                
                                # Get log probs for p_sum metrics (student and teacher probabilities)
                                student_log_probs = batch.batch.get("student_top_k_log_probs", None)  # (BS, SeqLen, K)
                                teacher_on_stu_log_probs = batch.batch.get("teacher_on_student_log_probs", None)  # (BS, SeqLen, K)
                                teacher_log_probs = batch.batch.get("teacher_top_k_log_probs", None)  # (BS, SeqLen, K)
                                student_on_tch_log_probs = batch.batch.get("student_log_probs_on_teacher_ids", None)  # (BS, SeqLen, K)

                                if top_k > 0 and advantages.dim() == 3:
                                    adv_k = advantages.shape[-1]  # K or 2K
                                    is_union = (strategy == "union" or strategy == "union-intersection") and (adv_k == 2 * top_k)
                                    
                                    # --- Global Metrics ---
                                    # Expand response mask to match advantages shape
                                    global_valid_mask_float = response_mask.unsqueeze(-1).expand(advantages.shape[0], advantages.shape[1], adv_k).float()
                                    global_valid_mask_bool = global_valid_mask_float > 0.5
                                    
                                    if is_union:
                                        # For union: front K is student, back K is teacher
                                        # overlap_mask: (B, T, K) - student id in teacher top k
                                        # teacher_in_student_mask: (B, T, K) - teacher id in student top k
                                        
                                        student_overlap = overlap_mask # (B, T, K)
                                        teacher_overlap = teacher_in_student_mask if teacher_in_student_mask is not None else torch.zeros_like(overlap_mask)
                                        
                                        # Build masks for the full 2K dimension
                                        # Front K: student top k
                                        #   - intersection: student_overlap > 0.5
                                        #   - only_stu: student_overlap < 0.5
                                        # Back K: teacher top k (only valid if not duplicate, i.e., ~teacher_overlap)
                                        #   - only_tch: ~teacher_overlap (teacher not in student)
                                        #   - (intersection on teacher side would be duplicate, already masked)
                                        
                                        student_adv = advantages[:, :, :top_k] # (B, T, K)
                                        teacher_adv = advantages[:, :, top_k:] # (B, T, K)
                                        
                                        student_valid = response_mask.unsqueeze(-1).expand_as(student_overlap).bool()
                                        teacher_valid = response_mask.unsqueeze(-1).expand_as(teacher_overlap).bool()
                                        
                                        # 1. Global Overlap Ratio (student side only for consistency)
                                        total_valid_k = student_valid.float().sum()
                                        total_overlap_k = (student_overlap * student_valid.float()).sum()
                                        
                                        if total_valid_k > 0:
                                            metrics["val-topk/overlap_ratio"] = (total_overlap_k / total_valid_k).item()
                                        
                                        # 2. Intersection Advantage (student tokens in teacher top k)
                                        mask_inter = (student_overlap > 0.5) & student_valid
                                        if mask_inter.any():
                                            avg_adv_inter = student_adv[mask_inter].mean()
                                            metrics["val-topk/adv_intersection"] = avg_adv_inter.item()
                                            
                                            # Compute p metrics for intersection (union strategy)
                                            if student_log_probs is not None and teacher_on_stu_log_probs is not None:
                                                student_p = torch.exp(student_log_probs)  # (B, T, K)
                                                teacher_p = torch.exp(teacher_on_stu_log_probs)  # (B, T, K)
                                                inter_positions = mask_inter.any(dim=-1)  # (B, T)
                                                
                                                # p_sum metrics
                                                student_p_masked = torch.where(mask_inter, student_p, torch.zeros_like(student_p))
                                                teacher_p_masked = torch.where(mask_inter, teacher_p, torch.zeros_like(teacher_p))
                                                student_p_sum = student_p_masked.sum(dim=-1)
                                                teacher_p_sum = teacher_p_masked.sum(dim=-1)
                                                metrics["val-topk/student_p_sum_intersection"] = student_p_sum[inter_positions].mean().item()
                                                metrics["val-topk/teacher_p_sum_intersection"] = teacher_p_sum[inter_positions].mean().item()
                                                
                                                # max_p metrics
                                                student_p_for_max = torch.where(mask_inter, student_p, torch.full_like(student_p, float('-inf')))
                                                teacher_p_for_max = torch.where(mask_inter, teacher_p, torch.full_like(teacher_p, float('-inf')))
                                                max_stu_idx = student_p_for_max.argmax(dim=-1)
                                                max_tch_idx = teacher_p_for_max.argmax(dim=-1)
                                                
                                                max_stu_p = student_p.gather(-1, max_stu_idx.unsqueeze(-1)).squeeze(-1)
                                                tch_p_at_max_stu = teacher_p.gather(-1, max_stu_idx.unsqueeze(-1)).squeeze(-1)
                                                adv_at_max_stu = student_adv.gather(-1, max_stu_idx.unsqueeze(-1)).squeeze(-1)
                                                max_tch_p = teacher_p.gather(-1, max_tch_idx.unsqueeze(-1)).squeeze(-1)
                                                stu_p_at_max_tch = student_p.gather(-1, max_tch_idx.unsqueeze(-1)).squeeze(-1)
                                                adv_at_max_tch = student_adv.gather(-1, max_tch_idx.unsqueeze(-1)).squeeze(-1)
                                                
                                                metrics["val-topk/max_student_p_intersection"] = max_stu_p[inter_positions].mean().item()
                                                metrics["val-topk/teacher_p_at_max_student_intersection"] = tch_p_at_max_stu[inter_positions].mean().item()
                                                metrics["val-topk/adv_at_max_student_intersection"] = adv_at_max_stu[inter_positions].mean().item()
                                                metrics["val-topk/max_teacher_p_intersection"] = max_tch_p[inter_positions].mean().item()
                                                metrics["val-topk/student_p_at_max_teacher_intersection"] = stu_p_at_max_tch[inter_positions].mean().item()
                                                metrics["val-topk/adv_at_max_teacher_intersection"] = adv_at_max_tch[inter_positions].mean().item()
                                                
                                                # max/min adv metrics
                                                adv_for_max = torch.where(mask_inter, student_adv, torch.full_like(student_adv, float('-inf')))
                                                adv_for_min = torch.where(mask_inter, student_adv, torch.full_like(student_adv, float('inf')))
                                                max_adv_idx = adv_for_max.argmax(dim=-1)
                                                min_adv_idx = adv_for_min.argmin(dim=-1)
                                                
                                                max_adv = student_adv.gather(-1, max_adv_idx.unsqueeze(-1)).squeeze(-1)
                                                stu_p_at_max_adv = student_p.gather(-1, max_adv_idx.unsqueeze(-1)).squeeze(-1)
                                                tch_p_at_max_adv = teacher_p.gather(-1, max_adv_idx.unsqueeze(-1)).squeeze(-1)
                                                min_adv = student_adv.gather(-1, min_adv_idx.unsqueeze(-1)).squeeze(-1)
                                                stu_p_at_min_adv = student_p.gather(-1, min_adv_idx.unsqueeze(-1)).squeeze(-1)
                                                tch_p_at_min_adv = teacher_p.gather(-1, min_adv_idx.unsqueeze(-1)).squeeze(-1)
                                                
                                                metrics["val-extrema/max_adv_intersection"] = max_adv[inter_positions].mean().item()
                                                metrics["val-extrema/student_p_at_max_adv_intersection"] = stu_p_at_max_adv[inter_positions].mean().item()
                                                metrics["val-extrema/teacher_p_at_max_adv_intersection"] = tch_p_at_max_adv[inter_positions].mean().item()
                                                metrics["val-extrema/min_adv_intersection"] = min_adv[inter_positions].mean().item()
                                                metrics["val-extrema/student_p_at_min_adv_intersection"] = stu_p_at_min_adv[inter_positions].mean().item()
                                                metrics["val-extrema/teacher_p_at_min_adv_intersection"] = tch_p_at_min_adv[inter_positions].mean().item()
                                        
                                        # 3. Only Student Advantage (student tokens NOT in teacher top k)
                                        mask_only_stu = (student_overlap < 0.5) & student_valid
                                        if mask_only_stu.any():
                                            avg_adv_only_stu = student_adv[mask_only_stu].mean()
                                            metrics["val-topk/adv_only_stu"] = avg_adv_only_stu.item()
                                            
                                            # val-extrema metrics for only_stu
                                            if student_log_probs is not None and teacher_on_stu_log_probs is not None:
                                                only_stu_positions = mask_only_stu.any(dim=-1)
                                                adv_for_max = torch.where(mask_only_stu, student_adv, torch.full_like(student_adv, float('-inf')))
                                                adv_for_min = torch.where(mask_only_stu, student_adv, torch.full_like(student_adv, float('inf')))
                                                max_adv_idx = adv_for_max.argmax(dim=-1)
                                                min_adv_idx = adv_for_min.argmin(dim=-1)
                                                
                                                max_adv = student_adv.gather(-1, max_adv_idx.unsqueeze(-1)).squeeze(-1)
                                                stu_p_at_max_adv = student_p.gather(-1, max_adv_idx.unsqueeze(-1)).squeeze(-1)
                                                tch_p_at_max_adv = teacher_p.gather(-1, max_adv_idx.unsqueeze(-1)).squeeze(-1)
                                                min_adv = student_adv.gather(-1, min_adv_idx.unsqueeze(-1)).squeeze(-1)
                                                stu_p_at_min_adv = student_p.gather(-1, min_adv_idx.unsqueeze(-1)).squeeze(-1)
                                                tch_p_at_min_adv = teacher_p.gather(-1, min_adv_idx.unsqueeze(-1)).squeeze(-1)
                                                
                                                metrics["val-extrema/max_adv_only_stu"] = max_adv[only_stu_positions].mean().item()
                                                metrics["val-extrema/student_p_at_max_adv_only_stu"] = stu_p_at_max_adv[only_stu_positions].mean().item()
                                                metrics["val-extrema/teacher_p_at_max_adv_only_stu"] = tch_p_at_max_adv[only_stu_positions].mean().item()
                                                metrics["val-extrema/min_adv_only_stu"] = min_adv[only_stu_positions].mean().item()
                                                metrics["val-extrema/student_p_at_min_adv_only_stu"] = stu_p_at_min_adv[only_stu_positions].mean().item()
                                                metrics["val-extrema/teacher_p_at_min_adv_only_stu"] = tch_p_at_min_adv[only_stu_positions].mean().item()
                                        
                                        # 4. Only Teacher Advantage (teacher tokens NOT in student top k)
                                        # These are the valid teacher tokens (not duplicated)
                                        mask_only_tch = (teacher_overlap < 0.5) & teacher_valid
                                        if mask_only_tch.any():
                                            avg_adv_only_tch = teacher_adv[mask_only_tch].mean()
                                            metrics["val-topk/adv_only_tch"] = avg_adv_only_tch.item()
                                            
                                            # val-extrema metrics for only_tch
                                            # For teacher tokens, we use teacher_adv and corresponding probabilities
                                            if teacher_log_probs is not None and student_on_tch_log_probs is not None:
                                                only_tch_positions = mask_only_tch.any(dim=-1)
                                                teacher_p_tch = torch.exp(teacher_log_probs)  # (B, T, K)
                                                student_p_tch = torch.exp(student_on_tch_log_probs)  # (B, T, K)
                                                
                                                adv_for_max = torch.where(mask_only_tch, teacher_adv, torch.full_like(teacher_adv, float('-inf')))
                                                adv_for_min = torch.where(mask_only_tch, teacher_adv, torch.full_like(teacher_adv, float('inf')))
                                                max_adv_idx = adv_for_max.argmax(dim=-1)
                                                min_adv_idx = adv_for_min.argmin(dim=-1)
                                                
                                                max_adv = teacher_adv.gather(-1, max_adv_idx.unsqueeze(-1)).squeeze(-1)
                                                stu_p_at_max_adv = student_p_tch.gather(-1, max_adv_idx.unsqueeze(-1)).squeeze(-1)
                                                tch_p_at_max_adv = teacher_p_tch.gather(-1, max_adv_idx.unsqueeze(-1)).squeeze(-1)
                                                min_adv = teacher_adv.gather(-1, min_adv_idx.unsqueeze(-1)).squeeze(-1)
                                                stu_p_at_min_adv = student_p_tch.gather(-1, min_adv_idx.unsqueeze(-1)).squeeze(-1)
                                                tch_p_at_min_adv = teacher_p_tch.gather(-1, min_adv_idx.unsqueeze(-1)).squeeze(-1)
                                                
                                                metrics["val-extrema/max_adv_only_tch"] = max_adv[only_tch_positions].mean().item()
                                                metrics["val-extrema/student_p_at_max_adv_only_tch"] = stu_p_at_max_adv[only_tch_positions].mean().item()
                                                metrics["val-extrema/teacher_p_at_max_adv_only_tch"] = tch_p_at_max_adv[only_tch_positions].mean().item()
                                                metrics["val-extrema/min_adv_only_tch"] = min_adv[only_tch_positions].mean().item()
                                                metrics["val-extrema/student_p_at_min_adv_only_tch"] = stu_p_at_min_adv[only_tch_positions].mean().item()
                                                metrics["val-extrema/teacher_p_at_min_adv_only_tch"] = tch_p_at_min_adv[only_tch_positions].mean().item()
                                        
                                        # --- Chunk-level metrics for union ---
                                        chunk_size = 1024
                                        for start_idx in range(0, max_len, chunk_size):
                                            end_idx = min(start_idx + chunk_size, max_len)
                                            chunk_key = f"{start_idx}_{end_idx}"
                                            
                                            chunk_response_mask = response_mask[:, start_idx:end_idx].bool()
                                            chunk_student_overlap = student_overlap[:, start_idx:end_idx]
                                            chunk_teacher_overlap = teacher_overlap[:, start_idx:end_idx]
                                            chunk_student_adv = student_adv[:, start_idx:end_idx]
                                            chunk_teacher_adv = teacher_adv[:, start_idx:end_idx]
                                            
                                            if not chunk_response_mask.any():
                                                continue
                                            
                                            chunk_student_valid = chunk_response_mask.unsqueeze(-1).expand_as(chunk_student_overlap)
                                            chunk_teacher_valid = chunk_response_mask.unsqueeze(-1).expand_as(chunk_teacher_overlap)
                                            
                                            # Overlap Ratio
                                            total_valid = chunk_student_valid.float().sum()
                                            total_overlap = (chunk_student_overlap * chunk_student_valid.float()).sum()
                                            if total_valid > 0:
                                                metrics[f"val-topk/overlap_ratio_chunk_{chunk_key}"] = (total_overlap / total_valid).item()
                                            
                                            # Intersection
                                            mask_inter_c = (chunk_student_overlap > 0.5) & chunk_student_valid
                                            if mask_inter_c.any():
                                                metrics[f"val-topk/adv_intersection_chunk_{chunk_key}"] = chunk_student_adv[mask_inter_c].mean().item()
                                                
                                                # Compute p metrics for intersection chunk (union strategy)
                                                if student_log_probs is not None and teacher_on_stu_log_probs is not None:
                                                    chunk_student_lp = student_log_probs[:, start_idx:end_idx]
                                                    chunk_teacher_lp = teacher_on_stu_log_probs[:, start_idx:end_idx]
                                                    student_p_c = torch.exp(chunk_student_lp)
                                                    teacher_p_c = torch.exp(chunk_teacher_lp)
                                                    inter_pos_c = mask_inter_c.any(dim=-1)
                                                    
                                                    # p_sum metrics
                                                    student_p_masked_c = torch.where(mask_inter_c, student_p_c, torch.zeros_like(student_p_c))
                                                    teacher_p_masked_c = torch.where(mask_inter_c, teacher_p_c, torch.zeros_like(teacher_p_c))
                                                    metrics[f"val-topk/student_p_sum_intersection_chunk_{chunk_key}"] = student_p_masked_c.sum(dim=-1)[inter_pos_c].mean().item()
                                                    metrics[f"val-topk/teacher_p_sum_intersection_chunk_{chunk_key}"] = teacher_p_masked_c.sum(dim=-1)[inter_pos_c].mean().item()
                                                    
                                                    # max_p metrics
                                                    student_p_for_max_c = torch.where(mask_inter_c, student_p_c, torch.full_like(student_p_c, float('-inf')))
                                                    teacher_p_for_max_c = torch.where(mask_inter_c, teacher_p_c, torch.full_like(teacher_p_c, float('-inf')))
                                                    max_stu_idx_c = student_p_for_max_c.argmax(dim=-1)
                                                    max_tch_idx_c = teacher_p_for_max_c.argmax(dim=-1)
                                                    
                                                    max_stu_p_c = student_p_c.gather(-1, max_stu_idx_c.unsqueeze(-1)).squeeze(-1)
                                                    tch_p_at_max_stu_c = teacher_p_c.gather(-1, max_stu_idx_c.unsqueeze(-1)).squeeze(-1)
                                                    adv_at_max_stu_c = chunk_student_adv.gather(-1, max_stu_idx_c.unsqueeze(-1)).squeeze(-1)
                                                    max_tch_p_c = teacher_p_c.gather(-1, max_tch_idx_c.unsqueeze(-1)).squeeze(-1)
                                                    stu_p_at_max_tch_c = student_p_c.gather(-1, max_tch_idx_c.unsqueeze(-1)).squeeze(-1)
                                                    adv_at_max_tch_c = chunk_student_adv.gather(-1, max_tch_idx_c.unsqueeze(-1)).squeeze(-1)
                                                    
                                                    metrics[f"val-topk/max_student_p_intersection_chunk_{chunk_key}"] = max_stu_p_c[inter_pos_c].mean().item()
                                                    metrics[f"val-topk/teacher_p_at_max_student_intersection_chunk_{chunk_key}"] = tch_p_at_max_stu_c[inter_pos_c].mean().item()
                                                    metrics[f"val-topk/adv_at_max_student_intersection_chunk_{chunk_key}"] = adv_at_max_stu_c[inter_pos_c].mean().item()
                                                    metrics[f"val-topk/max_teacher_p_intersection_chunk_{chunk_key}"] = max_tch_p_c[inter_pos_c].mean().item()
                                                    metrics[f"val-topk/student_p_at_max_teacher_intersection_chunk_{chunk_key}"] = stu_p_at_max_tch_c[inter_pos_c].mean().item()
                                                    metrics[f"val-topk/adv_at_max_teacher_intersection_chunk_{chunk_key}"] = adv_at_max_tch_c[inter_pos_c].mean().item()
                                                    
                                                    # max/min adv metrics
                                                    adv_for_max_c = torch.where(mask_inter_c, chunk_student_adv, torch.full_like(chunk_student_adv, float('-inf')))
                                                    adv_for_min_c = torch.where(mask_inter_c, chunk_student_adv, torch.full_like(chunk_student_adv, float('inf')))
                                                    max_adv_idx_c = adv_for_max_c.argmax(dim=-1)
                                                    min_adv_idx_c = adv_for_min_c.argmin(dim=-1)
                                                    
                                                    max_adv_c = chunk_student_adv.gather(-1, max_adv_idx_c.unsqueeze(-1)).squeeze(-1)
                                                    stu_p_at_max_adv_c = student_p_c.gather(-1, max_adv_idx_c.unsqueeze(-1)).squeeze(-1)
                                                    tch_p_at_max_adv_c = teacher_p_c.gather(-1, max_adv_idx_c.unsqueeze(-1)).squeeze(-1)
                                                    min_adv_c = chunk_student_adv.gather(-1, min_adv_idx_c.unsqueeze(-1)).squeeze(-1)
                                                    stu_p_at_min_adv_c = student_p_c.gather(-1, min_adv_idx_c.unsqueeze(-1)).squeeze(-1)
                                                    tch_p_at_min_adv_c = teacher_p_c.gather(-1, min_adv_idx_c.unsqueeze(-1)).squeeze(-1)
                                                    
                                                    metrics[f"val-extrema/max_adv_intersection_chunk_{chunk_key}"] = max_adv_c[inter_pos_c].mean().item()
                                                    metrics[f"val-extrema/student_p_at_max_adv_intersection_chunk_{chunk_key}"] = stu_p_at_max_adv_c[inter_pos_c].mean().item()
                                                    metrics[f"val-extrema/teacher_p_at_max_adv_intersection_chunk_{chunk_key}"] = tch_p_at_max_adv_c[inter_pos_c].mean().item()
                                                    metrics[f"val-extrema/min_adv_intersection_chunk_{chunk_key}"] = min_adv_c[inter_pos_c].mean().item()
                                                    metrics[f"val-extrema/student_p_at_min_adv_intersection_chunk_{chunk_key}"] = stu_p_at_min_adv_c[inter_pos_c].mean().item()
                                                    metrics[f"val-extrema/teacher_p_at_min_adv_intersection_chunk_{chunk_key}"] = tch_p_at_min_adv_c[inter_pos_c].mean().item()
                                            
                                            # Only Student
                                            mask_only_stu_c = (chunk_student_overlap < 0.5) & chunk_student_valid
                                            if mask_only_stu_c.any():
                                                metrics[f"val-topk/adv_only_stu_chunk_{chunk_key}"] = chunk_student_adv[mask_only_stu_c].mean().item()
                                                
                                                # val-extrema metrics for only_stu chunk
                                                if student_log_probs is not None and teacher_on_stu_log_probs is not None:
                                                    only_stu_pos_c = mask_only_stu_c.any(dim=-1)
                                                    adv_for_max_c = torch.where(mask_only_stu_c, chunk_student_adv, torch.full_like(chunk_student_adv, float('-inf')))
                                                    adv_for_min_c = torch.where(mask_only_stu_c, chunk_student_adv, torch.full_like(chunk_student_adv, float('inf')))
                                                    max_adv_idx_c = adv_for_max_c.argmax(dim=-1)
                                                    min_adv_idx_c = adv_for_min_c.argmin(dim=-1)
                                                    
                                                    max_adv_c = chunk_student_adv.gather(-1, max_adv_idx_c.unsqueeze(-1)).squeeze(-1)
                                                    stu_p_at_max_adv_c = student_p_c.gather(-1, max_adv_idx_c.unsqueeze(-1)).squeeze(-1)
                                                    tch_p_at_max_adv_c = teacher_p_c.gather(-1, max_adv_idx_c.unsqueeze(-1)).squeeze(-1)
                                                    min_adv_c = chunk_student_adv.gather(-1, min_adv_idx_c.unsqueeze(-1)).squeeze(-1)
                                                    stu_p_at_min_adv_c = student_p_c.gather(-1, min_adv_idx_c.unsqueeze(-1)).squeeze(-1)
                                                    tch_p_at_min_adv_c = teacher_p_c.gather(-1, min_adv_idx_c.unsqueeze(-1)).squeeze(-1)
                                                    
                                                    metrics[f"val-extrema/max_adv_only_stu_chunk_{chunk_key}"] = max_adv_c[only_stu_pos_c].mean().item()
                                                    metrics[f"val-extrema/student_p_at_max_adv_only_stu_chunk_{chunk_key}"] = stu_p_at_max_adv_c[only_stu_pos_c].mean().item()
                                                    metrics[f"val-extrema/teacher_p_at_max_adv_only_stu_chunk_{chunk_key}"] = tch_p_at_max_adv_c[only_stu_pos_c].mean().item()
                                                    metrics[f"val-extrema/min_adv_only_stu_chunk_{chunk_key}"] = min_adv_c[only_stu_pos_c].mean().item()
                                                    metrics[f"val-extrema/student_p_at_min_adv_only_stu_chunk_{chunk_key}"] = stu_p_at_min_adv_c[only_stu_pos_c].mean().item()
                                                    metrics[f"val-extrema/teacher_p_at_min_adv_only_stu_chunk_{chunk_key}"] = tch_p_at_min_adv_c[only_stu_pos_c].mean().item()
                                            
                                            # Only Teacher
                                            mask_only_tch_c = (chunk_teacher_overlap < 0.5) & chunk_teacher_valid
                                            if mask_only_tch_c.any():
                                                metrics[f"val-topk/adv_only_tch_chunk_{chunk_key}"] = chunk_teacher_adv[mask_only_tch_c].mean().item()
                                                
                                                # val-extrema metrics for only_tch chunk
                                                if teacher_log_probs is not None and student_on_tch_log_probs is not None:
                                                    only_tch_pos_c = mask_only_tch_c.any(dim=-1)
                                                    chunk_teacher_lp = teacher_log_probs[:, start_idx:end_idx]
                                                    chunk_stu_on_tch_lp = student_on_tch_log_probs[:, start_idx:end_idx]
                                                    teacher_p_tch_c = torch.exp(chunk_teacher_lp)
                                                    student_p_tch_c = torch.exp(chunk_stu_on_tch_lp)
                                                    
                                                    adv_for_max_c = torch.where(mask_only_tch_c, chunk_teacher_adv, torch.full_like(chunk_teacher_adv, float('-inf')))
                                                    adv_for_min_c = torch.where(mask_only_tch_c, chunk_teacher_adv, torch.full_like(chunk_teacher_adv, float('inf')))
                                                    max_adv_idx_c = adv_for_max_c.argmax(dim=-1)
                                                    min_adv_idx_c = adv_for_min_c.argmin(dim=-1)
                                                    
                                                    max_adv_c = chunk_teacher_adv.gather(-1, max_adv_idx_c.unsqueeze(-1)).squeeze(-1)
                                                    stu_p_at_max_adv_c = student_p_tch_c.gather(-1, max_adv_idx_c.unsqueeze(-1)).squeeze(-1)
                                                    tch_p_at_max_adv_c = teacher_p_tch_c.gather(-1, max_adv_idx_c.unsqueeze(-1)).squeeze(-1)
                                                    min_adv_c = chunk_teacher_adv.gather(-1, min_adv_idx_c.unsqueeze(-1)).squeeze(-1)
                                                    stu_p_at_min_adv_c = student_p_tch_c.gather(-1, min_adv_idx_c.unsqueeze(-1)).squeeze(-1)
                                                    tch_p_at_min_adv_c = teacher_p_tch_c.gather(-1, min_adv_idx_c.unsqueeze(-1)).squeeze(-1)
                                                    
                                                    metrics[f"val-extrema/max_adv_only_tch_chunk_{chunk_key}"] = max_adv_c[only_tch_pos_c].mean().item()
                                                    metrics[f"val-extrema/student_p_at_max_adv_only_tch_chunk_{chunk_key}"] = stu_p_at_max_adv_c[only_tch_pos_c].mean().item()
                                                    metrics[f"val-extrema/teacher_p_at_max_adv_only_tch_chunk_{chunk_key}"] = tch_p_at_max_adv_c[only_tch_pos_c].mean().item()
                                                    metrics[f"val-extrema/min_adv_only_tch_chunk_{chunk_key}"] = min_adv_c[only_tch_pos_c].mean().item()
                                                    metrics[f"val-extrema/student_p_at_min_adv_only_tch_chunk_{chunk_key}"] = stu_p_at_min_adv_c[only_tch_pos_c].mean().item()
                                                    metrics[f"val-extrema/teacher_p_at_min_adv_only_tch_chunk_{chunk_key}"] = tch_p_at_min_adv_c[only_tch_pos_c].mean().item()
                                    
                                    else:
                                        # Non-union strategies (only_stu, only_tch, intersection)
                                        # For only_tch, use teacher_in_student_mask; for others, use overlap_mask
                                        if strategy == "only_tch" and "teacher_in_student_mask" in batch.batch:
                                            # For only_tch: advantages are for Teacher top k
                                            # teacher_in_student_mask: (B, T, K) - Teacher ID in Student top k
                                            tch_in_stu_mask = batch.batch["teacher_in_student_mask"]
                                            global_valid_mask_float_k = response_mask.unsqueeze(-1).expand_as(tch_in_stu_mask).float()
                                            global_valid_mask_bool_k = global_valid_mask_float_k > 0.5
                                            
                                            # 1. Global Overlap Ratio (Teacher side)
                                            global_total_valid_k = global_valid_mask_float_k.sum()
                                            global_total_overlap_k = (tch_in_stu_mask * global_valid_mask_float_k).sum()
                                            
                                            if global_total_valid_k > 0:
                                                metrics["val-topk/overlap_ratio"] = (global_total_overlap_k / global_total_valid_k).item()
                                            
                                            # 2. Intersection Advantage (Teacher tokens in Student top k)
                                            global_mask_inter = (tch_in_stu_mask > 0.5) & global_valid_mask_bool_k
                                            if global_mask_inter.any():
                                                global_avg_adv_inter = advantages[global_mask_inter].mean()
                                                metrics["val-topk/adv_intersection"] = global_avg_adv_inter.item()
                                                
                                                # Compute p metrics for intersection (only_tch strategy)
                                                student_on_tch_log_probs = batch.batch.get("student_log_probs_on_teacher_ids", None)
                                                teacher_top_k_lp = batch.batch.get("teacher_top_k_log_probs", None)
                                                if student_on_tch_log_probs is not None and teacher_top_k_lp is not None:
                                                    student_p = torch.exp(student_on_tch_log_probs)
                                                    teacher_p = torch.exp(teacher_top_k_lp)
                                                    inter_positions = global_mask_inter.any(dim=-1)
                                                    
                                                    # p_sum metrics
                                                    student_p_masked = torch.where(global_mask_inter, student_p, torch.zeros_like(student_p))
                                                    teacher_p_masked = torch.where(global_mask_inter, teacher_p, torch.zeros_like(teacher_p))
                                                    metrics["val-topk/student_p_sum_intersection"] = student_p_masked.sum(dim=-1)[inter_positions].mean().item()
                                                    metrics["val-topk/teacher_p_sum_intersection"] = teacher_p_masked.sum(dim=-1)[inter_positions].mean().item()
                                                    
                                                    # max_p metrics
                                                    student_p_for_max = torch.where(global_mask_inter, student_p, torch.full_like(student_p, float('-inf')))
                                                    teacher_p_for_max = torch.where(global_mask_inter, teacher_p, torch.full_like(teacher_p, float('-inf')))
                                                    max_stu_idx = student_p_for_max.argmax(dim=-1)
                                                    max_tch_idx = teacher_p_for_max.argmax(dim=-1)
                                                    
                                                    max_stu_p = student_p.gather(-1, max_stu_idx.unsqueeze(-1)).squeeze(-1)
                                                    tch_p_at_max_stu = teacher_p.gather(-1, max_stu_idx.unsqueeze(-1)).squeeze(-1)
                                                    adv_at_max_stu = advantages.gather(-1, max_stu_idx.unsqueeze(-1)).squeeze(-1)
                                                    max_tch_p = teacher_p.gather(-1, max_tch_idx.unsqueeze(-1)).squeeze(-1)
                                                    stu_p_at_max_tch = student_p.gather(-1, max_tch_idx.unsqueeze(-1)).squeeze(-1)
                                                    adv_at_max_tch = advantages.gather(-1, max_tch_idx.unsqueeze(-1)).squeeze(-1)
                                                    
                                                    metrics["val-topk/max_student_p_intersection"] = max_stu_p[inter_positions].mean().item()
                                                    metrics["val-topk/teacher_p_at_max_student_intersection"] = tch_p_at_max_stu[inter_positions].mean().item()
                                                    metrics["val-topk/adv_at_max_student_intersection"] = adv_at_max_stu[inter_positions].mean().item()
                                                    metrics["val-topk/max_teacher_p_intersection"] = max_tch_p[inter_positions].mean().item()
                                                    metrics["val-topk/student_p_at_max_teacher_intersection"] = stu_p_at_max_tch[inter_positions].mean().item()
                                                    metrics["val-topk/adv_at_max_teacher_intersection"] = adv_at_max_tch[inter_positions].mean().item()
                                                    
                                                    # max/min adv metrics
                                                    adv_for_max = torch.where(global_mask_inter, advantages, torch.full_like(advantages, float('-inf')))
                                                    adv_for_min = torch.where(global_mask_inter, advantages, torch.full_like(advantages, float('inf')))
                                                    max_adv_idx = adv_for_max.argmax(dim=-1)
                                                    min_adv_idx = adv_for_min.argmin(dim=-1)
                                                    
                                                    max_adv = advantages.gather(-1, max_adv_idx.unsqueeze(-1)).squeeze(-1)
                                                    stu_p_at_max_adv = student_p.gather(-1, max_adv_idx.unsqueeze(-1)).squeeze(-1)
                                                    tch_p_at_max_adv = teacher_p.gather(-1, max_adv_idx.unsqueeze(-1)).squeeze(-1)
                                                    min_adv = advantages.gather(-1, min_adv_idx.unsqueeze(-1)).squeeze(-1)
                                                    stu_p_at_min_adv = student_p.gather(-1, min_adv_idx.unsqueeze(-1)).squeeze(-1)
                                                    tch_p_at_min_adv = teacher_p.gather(-1, min_adv_idx.unsqueeze(-1)).squeeze(-1)
                                                    
                                                    metrics["val-extrema/max_adv_intersection"] = max_adv[inter_positions].mean().item()
                                                    metrics["val-extrema/student_p_at_max_adv_intersection"] = stu_p_at_max_adv[inter_positions].mean().item()
                                                    metrics["val-extrema/teacher_p_at_max_adv_intersection"] = tch_p_at_max_adv[inter_positions].mean().item()
                                                    metrics["val-extrema/min_adv_intersection"] = min_adv[inter_positions].mean().item()
                                                    metrics["val-extrema/student_p_at_min_adv_intersection"] = stu_p_at_min_adv[inter_positions].mean().item()
                                                    metrics["val-extrema/teacher_p_at_min_adv_intersection"] = tch_p_at_min_adv[inter_positions].mean().item()
                                                
                                            # 3. Only Teacher Advantage (Teacher tokens NOT in Student top k)
                                            global_mask_only_tch = (tch_in_stu_mask < 0.5) & global_valid_mask_bool_k
                                            if global_mask_only_tch.any():
                                                global_avg_adv_only_tch = advantages[global_mask_only_tch].mean()
                                                metrics["val-topk/adv_only_tch"] = global_avg_adv_only_tch.item()

                                            chunk_size = 1024
                                            for start_idx in range(0, max_len, chunk_size):
                                                end_idx = min(start_idx + chunk_size, max_len)
                                                chunk_key = f"{start_idx}_{end_idx}"
                                                
                                                chunk_response_mask = response_mask[:, start_idx:end_idx].bool()
                                                chunk_tch_in_stu = tch_in_stu_mask[:, start_idx:end_idx]
                                                chunk_adv = advantages[:, start_idx:end_idx]
                                                
                                                if not chunk_response_mask.any():
                                                    continue
                                                
                                                chunk_valid_mask = chunk_response_mask.unsqueeze(-1).expand_as(chunk_tch_in_stu)
                                                
                                                # Overlap Ratio
                                                total_valid_k = chunk_valid_mask.sum()
                                                total_overlap_k = (chunk_tch_in_stu * chunk_valid_mask.float()).sum()
                                                if total_valid_k > 0:
                                                    metrics[f"val-topk/overlap_ratio_chunk_{chunk_key}"] = (total_overlap_k / total_valid_k).item()
                                                
                                                # Intersection
                                                mask_inter = (chunk_tch_in_stu > 0.5) & chunk_valid_mask
                                                if mask_inter.any():
                                                    metrics[f"val-topk/adv_intersection_chunk_{chunk_key}"] = chunk_adv[mask_inter].mean().item()
                                                    
                                                    # Compute p metrics for intersection chunk (only_tch strategy)
                                                    if student_on_tch_log_probs is not None and teacher_top_k_lp is not None:
                                                        chunk_stu_lp = student_on_tch_log_probs[:, start_idx:end_idx]
                                                        chunk_tch_lp = teacher_top_k_lp[:, start_idx:end_idx]
                                                        student_p_c = torch.exp(chunk_stu_lp)
                                                        teacher_p_c = torch.exp(chunk_tch_lp)
                                                        inter_pos_c = mask_inter.any(dim=-1)
                                                        
                                                        # p_sum metrics
                                                        student_p_masked_c = torch.where(mask_inter, student_p_c, torch.zeros_like(student_p_c))
                                                        teacher_p_masked_c = torch.where(mask_inter, teacher_p_c, torch.zeros_like(teacher_p_c))
                                                        metrics[f"val-topk/student_p_sum_intersection_chunk_{chunk_key}"] = student_p_masked_c.sum(dim=-1)[inter_pos_c].mean().item()
                                                        metrics[f"val-topk/teacher_p_sum_intersection_chunk_{chunk_key}"] = teacher_p_masked_c.sum(dim=-1)[inter_pos_c].mean().item()
                                                        
                                                        # max_p metrics
                                                        student_p_for_max_c = torch.where(mask_inter, student_p_c, torch.full_like(student_p_c, float('-inf')))
                                                        teacher_p_for_max_c = torch.where(mask_inter, teacher_p_c, torch.full_like(teacher_p_c, float('-inf')))
                                                        max_stu_idx_c = student_p_for_max_c.argmax(dim=-1)
                                                        max_tch_idx_c = teacher_p_for_max_c.argmax(dim=-1)
                                                        
                                                        max_stu_p_c = student_p_c.gather(-1, max_stu_idx_c.unsqueeze(-1)).squeeze(-1)
                                                        tch_p_at_max_stu_c = teacher_p_c.gather(-1, max_stu_idx_c.unsqueeze(-1)).squeeze(-1)
                                                        adv_at_max_stu_c = chunk_adv.gather(-1, max_stu_idx_c.unsqueeze(-1)).squeeze(-1)
                                                        max_tch_p_c = teacher_p_c.gather(-1, max_tch_idx_c.unsqueeze(-1)).squeeze(-1)
                                                        stu_p_at_max_tch_c = student_p_c.gather(-1, max_tch_idx_c.unsqueeze(-1)).squeeze(-1)
                                                        adv_at_max_tch_c = chunk_adv.gather(-1, max_tch_idx_c.unsqueeze(-1)).squeeze(-1)
                                                        
                                                        metrics[f"val-topk/max_student_p_intersection_chunk_{chunk_key}"] = max_stu_p_c[inter_pos_c].mean().item()
                                                        metrics[f"val-topk/teacher_p_at_max_student_intersection_chunk_{chunk_key}"] = tch_p_at_max_stu_c[inter_pos_c].mean().item()
                                                        metrics[f"val-topk/adv_at_max_student_intersection_chunk_{chunk_key}"] = adv_at_max_stu_c[inter_pos_c].mean().item()
                                                        metrics[f"val-topk/max_teacher_p_intersection_chunk_{chunk_key}"] = max_tch_p_c[inter_pos_c].mean().item()
                                                        metrics[f"val-topk/student_p_at_max_teacher_intersection_chunk_{chunk_key}"] = stu_p_at_max_tch_c[inter_pos_c].mean().item()
                                                        metrics[f"val-topk/adv_at_max_teacher_intersection_chunk_{chunk_key}"] = adv_at_max_tch_c[inter_pos_c].mean().item()
                                                        
                                                        # max/min adv metrics
                                                        adv_for_max_c = torch.where(mask_inter, chunk_adv, torch.full_like(chunk_adv, float('-inf')))
                                                        adv_for_min_c = torch.where(mask_inter, chunk_adv, torch.full_like(chunk_adv, float('inf')))
                                                        max_adv_idx_c = adv_for_max_c.argmax(dim=-1)
                                                        min_adv_idx_c = adv_for_min_c.argmin(dim=-1)
                                                        
                                                        max_adv_c = chunk_adv.gather(-1, max_adv_idx_c.unsqueeze(-1)).squeeze(-1)
                                                        stu_p_at_max_adv_c = student_p_c.gather(-1, max_adv_idx_c.unsqueeze(-1)).squeeze(-1)
                                                        tch_p_at_max_adv_c = teacher_p_c.gather(-1, max_adv_idx_c.unsqueeze(-1)).squeeze(-1)
                                                        min_adv_c = chunk_adv.gather(-1, min_adv_idx_c.unsqueeze(-1)).squeeze(-1)
                                                        stu_p_at_min_adv_c = student_p_c.gather(-1, min_adv_idx_c.unsqueeze(-1)).squeeze(-1)
                                                        tch_p_at_min_adv_c = teacher_p_c.gather(-1, min_adv_idx_c.unsqueeze(-1)).squeeze(-1)
                                                        
                                                        metrics[f"val-extrema/max_adv_intersection_chunk_{chunk_key}"] = max_adv_c[inter_pos_c].mean().item()
                                                        metrics[f"val-extrema/student_p_at_max_adv_intersection_chunk_{chunk_key}"] = stu_p_at_max_adv_c[inter_pos_c].mean().item()
                                                        metrics[f"val-extrema/teacher_p_at_max_adv_intersection_chunk_{chunk_key}"] = tch_p_at_max_adv_c[inter_pos_c].mean().item()
                                                        metrics[f"val-extrema/min_adv_intersection_chunk_{chunk_key}"] = min_adv_c[inter_pos_c].mean().item()
                                                        metrics[f"val-extrema/student_p_at_min_adv_intersection_chunk_{chunk_key}"] = stu_p_at_min_adv_c[inter_pos_c].mean().item()
                                                        metrics[f"val-extrema/teacher_p_at_min_adv_intersection_chunk_{chunk_key}"] = tch_p_at_min_adv_c[inter_pos_c].mean().item()
                                                
                                                # Only Teacher
                                                mask_only_tch = (chunk_tch_in_stu < 0.5) & chunk_valid_mask
                                                if mask_only_tch.any():
                                                    metrics[f"val-topk/adv_only_tch_chunk_{chunk_key}"] = chunk_adv[mask_only_tch].mean().item()
                                        else:
                                            # only_stu, intersection: overlap_mask and advantages both (B, T, K)
                                            global_valid_mask_float_k = response_mask.unsqueeze(-1).expand_as(overlap_mask).float()
                                            global_valid_mask_bool_k = global_valid_mask_float_k > 0.5
                                            
                                            # 1. Global Overlap Ratio
                                            global_total_valid_k = global_valid_mask_float_k.sum()
                                            global_total_overlap_k = (overlap_mask * global_valid_mask_float_k).sum()
                                            
                                            if global_total_valid_k > 0:
                                                metrics["val-topk/overlap_ratio"] = (global_total_overlap_k / global_total_valid_k).item()
                                            
                                            # 2. Global Advantage Analysis
                                            # Intersection Advantage
                                            global_mask_inter = (overlap_mask > 0.5) & global_valid_mask_bool_k
                                            if global_mask_inter.any():
                                                global_avg_adv_inter = advantages[global_mask_inter].mean()
                                                metrics["val-topk/adv_intersection"] = global_avg_adv_inter.item()
                                                
                                                # Compute p metrics for intersection (only_stu/intersection strategy)
                                                if student_log_probs is not None and teacher_on_stu_log_probs is not None:
                                                    student_p = torch.exp(student_log_probs)
                                                    teacher_p = torch.exp(teacher_on_stu_log_probs)
                                                    inter_positions = global_mask_inter.any(dim=-1)
                                                    
                                                    # p_sum metrics
                                                    student_p_masked = torch.where(global_mask_inter, student_p, torch.zeros_like(student_p))
                                                    teacher_p_masked = torch.where(global_mask_inter, teacher_p, torch.zeros_like(teacher_p))
                                                    metrics["val-topk/student_p_sum_intersection"] = student_p_masked.sum(dim=-1)[inter_positions].mean().item()
                                                    metrics["val-topk/teacher_p_sum_intersection"] = teacher_p_masked.sum(dim=-1)[inter_positions].mean().item()
                                                    
                                                    # max_p metrics
                                                    student_p_for_max = torch.where(global_mask_inter, student_p, torch.full_like(student_p, float('-inf')))
                                                    teacher_p_for_max = torch.where(global_mask_inter, teacher_p, torch.full_like(teacher_p, float('-inf')))
                                                    max_stu_idx = student_p_for_max.argmax(dim=-1)
                                                    max_tch_idx = teacher_p_for_max.argmax(dim=-1)
                                                    
                                                    max_stu_p = student_p.gather(-1, max_stu_idx.unsqueeze(-1)).squeeze(-1)
                                                    tch_p_at_max_stu = teacher_p.gather(-1, max_stu_idx.unsqueeze(-1)).squeeze(-1)
                                                    adv_at_max_stu = advantages.gather(-1, max_stu_idx.unsqueeze(-1)).squeeze(-1)
                                                    max_tch_p = teacher_p.gather(-1, max_tch_idx.unsqueeze(-1)).squeeze(-1)
                                                    stu_p_at_max_tch = student_p.gather(-1, max_tch_idx.unsqueeze(-1)).squeeze(-1)
                                                    adv_at_max_tch = advantages.gather(-1, max_tch_idx.unsqueeze(-1)).squeeze(-1)
                                                    
                                                    metrics["val-topk/max_student_p_intersection"] = max_stu_p[inter_positions].mean().item()
                                                    metrics["val-topk/teacher_p_at_max_student_intersection"] = tch_p_at_max_stu[inter_positions].mean().item()
                                                    metrics["val-topk/adv_at_max_student_intersection"] = adv_at_max_stu[inter_positions].mean().item()
                                                    metrics["val-topk/max_teacher_p_intersection"] = max_tch_p[inter_positions].mean().item()
                                                    metrics["val-topk/student_p_at_max_teacher_intersection"] = stu_p_at_max_tch[inter_positions].mean().item()
                                                    metrics["val-topk/adv_at_max_teacher_intersection"] = adv_at_max_tch[inter_positions].mean().item()
                                                    
                                                    # max/min adv metrics
                                                    adv_for_max = torch.where(global_mask_inter, advantages, torch.full_like(advantages, float('-inf')))
                                                    adv_for_min = torch.where(global_mask_inter, advantages, torch.full_like(advantages, float('inf')))
                                                    max_adv_idx = adv_for_max.argmax(dim=-1)
                                                    min_adv_idx = adv_for_min.argmin(dim=-1)
                                                    
                                                    max_adv = advantages.gather(-1, max_adv_idx.unsqueeze(-1)).squeeze(-1)
                                                    stu_p_at_max_adv = student_p.gather(-1, max_adv_idx.unsqueeze(-1)).squeeze(-1)
                                                    tch_p_at_max_adv = teacher_p.gather(-1, max_adv_idx.unsqueeze(-1)).squeeze(-1)
                                                    min_adv = advantages.gather(-1, min_adv_idx.unsqueeze(-1)).squeeze(-1)
                                                    stu_p_at_min_adv = student_p.gather(-1, min_adv_idx.unsqueeze(-1)).squeeze(-1)
                                                    tch_p_at_min_adv = teacher_p.gather(-1, min_adv_idx.unsqueeze(-1)).squeeze(-1)
                                                    
                                                    metrics["val-extrema/max_adv_intersection"] = max_adv[inter_positions].mean().item()
                                                    metrics["val-extrema/student_p_at_max_adv_intersection"] = stu_p_at_max_adv[inter_positions].mean().item()
                                                    metrics["val-extrema/teacher_p_at_max_adv_intersection"] = tch_p_at_max_adv[inter_positions].mean().item()
                                                    metrics["val-extrema/min_adv_intersection"] = min_adv[inter_positions].mean().item()
                                                    metrics["val-extrema/student_p_at_min_adv_intersection"] = stu_p_at_min_adv[inter_positions].mean().item()
                                                    metrics["val-extrema/teacher_p_at_min_adv_intersection"] = tch_p_at_min_adv[inter_positions].mean().item()
                                                
                                            # Only Student Advantage
                                            global_mask_only_stu = (overlap_mask < 0.5) & global_valid_mask_bool_k
                                            if global_mask_only_stu.any():
                                                global_avg_adv_only_stu = advantages[global_mask_only_stu].mean()
                                                metrics["val-topk/adv_only_stu"] = global_avg_adv_only_stu.item()

                                            chunk_size = 1024
                                            
                                            # We can iterate up to max_len
                                            for start_idx in range(0, max_len, chunk_size):
                                                end_idx = min(start_idx + chunk_size, max_len)
                                                chunk_key = f"{start_idx}_{end_idx}"
                                                
                                                # Slice tensors
                                                chunk_response_mask = response_mask[:, start_idx:end_idx].bool() # (BS, Chunk)
                                                chunk_overlap_mask = overlap_mask[:, start_idx:end_idx] # (BS, Chunk, K)
                                                chunk_adv = advantages[:, start_idx:end_idx] # (BS, Chunk, K)
                                                
                                                if not chunk_response_mask.any():
                                                    continue
                                                
                                                # Expand response mask to K for element-wise ops
                                                chunk_valid_mask = chunk_response_mask.unsqueeze(-1).expand_as(chunk_overlap_mask)
                                                
                                                # 1. Overlap Ratio per chunk
                                                total_valid_k = chunk_valid_mask.sum()
                                                total_overlap_k = (chunk_overlap_mask * chunk_valid_mask.float()).sum()
                                                
                                                if total_valid_k > 0:
                                                    metrics[f"val-topk/overlap_ratio_chunk_{chunk_key}"] = (total_overlap_k / total_valid_k).item()
                                                
                                                # 2. Advantage Analysis
                                                # Intersection Advantage
                                                mask_inter = (chunk_overlap_mask > 0.5) & chunk_valid_mask
                                                if mask_inter.any():
                                                    avg_adv_inter = chunk_adv[mask_inter].mean()
                                                    metrics[f"val-topk/adv_intersection_chunk_{chunk_key}"] = avg_adv_inter.item()
                                                    
                                                    # Compute p metrics for intersection chunk (only_stu/intersection strategy)
                                                    if student_log_probs is not None and teacher_on_stu_log_probs is not None:
                                                        chunk_student_lp = student_log_probs[:, start_idx:end_idx]
                                                        chunk_teacher_lp = teacher_on_stu_log_probs[:, start_idx:end_idx]
                                                        student_p_c = torch.exp(chunk_student_lp)
                                                        teacher_p_c = torch.exp(chunk_teacher_lp)
                                                        inter_pos_c = mask_inter.any(dim=-1)
                                                        
                                                        # p_sum metrics
                                                        student_p_masked_c = torch.where(mask_inter, student_p_c, torch.zeros_like(student_p_c))
                                                        teacher_p_masked_c = torch.where(mask_inter, teacher_p_c, torch.zeros_like(teacher_p_c))
                                                        metrics[f"val-topk/student_p_sum_intersection_chunk_{chunk_key}"] = student_p_masked_c.sum(dim=-1)[inter_pos_c].mean().item()
                                                        metrics[f"val-topk/teacher_p_sum_intersection_chunk_{chunk_key}"] = teacher_p_masked_c.sum(dim=-1)[inter_pos_c].mean().item()
                                                        
                                                        # max_p metrics
                                                        student_p_for_max_c = torch.where(mask_inter, student_p_c, torch.full_like(student_p_c, float('-inf')))
                                                        teacher_p_for_max_c = torch.where(mask_inter, teacher_p_c, torch.full_like(teacher_p_c, float('-inf')))
                                                        max_stu_idx_c = student_p_for_max_c.argmax(dim=-1)
                                                        max_tch_idx_c = teacher_p_for_max_c.argmax(dim=-1)
                                                        
                                                        max_stu_p_c = student_p_c.gather(-1, max_stu_idx_c.unsqueeze(-1)).squeeze(-1)
                                                        tch_p_at_max_stu_c = teacher_p_c.gather(-1, max_stu_idx_c.unsqueeze(-1)).squeeze(-1)
                                                        adv_at_max_stu_c = chunk_adv.gather(-1, max_stu_idx_c.unsqueeze(-1)).squeeze(-1)
                                                        max_tch_p_c = teacher_p_c.gather(-1, max_tch_idx_c.unsqueeze(-1)).squeeze(-1)
                                                        stu_p_at_max_tch_c = student_p_c.gather(-1, max_tch_idx_c.unsqueeze(-1)).squeeze(-1)
                                                        adv_at_max_tch_c = chunk_adv.gather(-1, max_tch_idx_c.unsqueeze(-1)).squeeze(-1)
                                                        
                                                        metrics[f"val-topk/max_student_p_intersection_chunk_{chunk_key}"] = max_stu_p_c[inter_pos_c].mean().item()
                                                        metrics[f"val-topk/teacher_p_at_max_student_intersection_chunk_{chunk_key}"] = tch_p_at_max_stu_c[inter_pos_c].mean().item()
                                                        metrics[f"val-topk/adv_at_max_student_intersection_chunk_{chunk_key}"] = adv_at_max_stu_c[inter_pos_c].mean().item()
                                                        metrics[f"val-topk/max_teacher_p_intersection_chunk_{chunk_key}"] = max_tch_p_c[inter_pos_c].mean().item()
                                                        metrics[f"val-topk/student_p_at_max_teacher_intersection_chunk_{chunk_key}"] = stu_p_at_max_tch_c[inter_pos_c].mean().item()
                                                        metrics[f"val-topk/adv_at_max_teacher_intersection_chunk_{chunk_key}"] = adv_at_max_tch_c[inter_pos_c].mean().item()
                                                        
                                                        # max/min adv metrics
                                                        adv_for_max_c = torch.where(mask_inter, chunk_adv, torch.full_like(chunk_adv, float('-inf')))
                                                        adv_for_min_c = torch.where(mask_inter, chunk_adv, torch.full_like(chunk_adv, float('inf')))
                                                        max_adv_idx_c = adv_for_max_c.argmax(dim=-1)
                                                        min_adv_idx_c = adv_for_min_c.argmin(dim=-1)
                                                        
                                                        max_adv_c = chunk_adv.gather(-1, max_adv_idx_c.unsqueeze(-1)).squeeze(-1)
                                                        stu_p_at_max_adv_c = student_p_c.gather(-1, max_adv_idx_c.unsqueeze(-1)).squeeze(-1)
                                                        tch_p_at_max_adv_c = teacher_p_c.gather(-1, max_adv_idx_c.unsqueeze(-1)).squeeze(-1)
                                                        min_adv_c = chunk_adv.gather(-1, min_adv_idx_c.unsqueeze(-1)).squeeze(-1)
                                                        stu_p_at_min_adv_c = student_p_c.gather(-1, min_adv_idx_c.unsqueeze(-1)).squeeze(-1)
                                                        tch_p_at_min_adv_c = teacher_p_c.gather(-1, min_adv_idx_c.unsqueeze(-1)).squeeze(-1)
                                                        
                                                        metrics[f"val-extrema/max_adv_intersection_chunk_{chunk_key}"] = max_adv_c[inter_pos_c].mean().item()
                                                        metrics[f"val-extrema/student_p_at_max_adv_intersection_chunk_{chunk_key}"] = stu_p_at_max_adv_c[inter_pos_c].mean().item()
                                                        metrics[f"val-extrema/teacher_p_at_max_adv_intersection_chunk_{chunk_key}"] = tch_p_at_max_adv_c[inter_pos_c].mean().item()
                                                        metrics[f"val-extrema/min_adv_intersection_chunk_{chunk_key}"] = min_adv_c[inter_pos_c].mean().item()
                                                        metrics[f"val-extrema/student_p_at_min_adv_intersection_chunk_{chunk_key}"] = stu_p_at_min_adv_c[inter_pos_c].mean().item()
                                                        metrics[f"val-extrema/teacher_p_at_min_adv_intersection_chunk_{chunk_key}"] = tch_p_at_min_adv_c[inter_pos_c].mean().item()
                                                    
                                                # Only Student Advantage
                                                mask_only_stu = (chunk_overlap_mask < 0.5) & chunk_valid_mask
                                                if mask_only_stu.any():
                                                    avg_adv_only_stu = chunk_adv[mask_only_stu].mean()
                                                    metrics[f"val-topk/adv_only_stu_chunk_{chunk_key}"] = avg_adv_only_stu.item()
                                            
                            except Exception as e:
                                print(f"Error computing Top-K metrics: {e}")
                                import traceback
                                traceback.print_exc()
                    
                    if self.config.trainer.get("is_plot", False) and (self.global_steps == 1 or self.global_steps % 10 == 0):
                        try:
                            import matplotlib.pyplot as plt
                            import swanlab
                            
                            # Check if teacher_entropy is available
                            if "teacher_entropy" in batch.batch.keys():
                                teacher_entropy = batch.batch["teacher_entropy"]
                                
                                # Determine advantage to use
                                if "token_level_advantage_direct" in batch.batch.keys():
                                    adv = batch.batch["token_level_advantage_direct"]
                                else:
                                    adv = batch.batch["advantages"]

                                if adv.dim() == 3:
                                    adv = adv.sum(dim=-1)
                                
                                response_mask = batch.batch["response_mask"]
                                
                                # Move to CPU and detach
                                teacher_entropy_cpu = teacher_entropy.detach().cpu()
                                adv_cpu = adv.detach().cpu()
                                mask_cpu = response_mask.detach().cpu().bool()
                                
                                # Create position indices
                                batch_size, seq_len = teacher_entropy_cpu.shape
                                positions = torch.arange(seq_len).unsqueeze(0).expand(batch_size, seq_len)
                                
                                # Filter using mask
                                valid_indices = mask_cpu
                                valid_positions = positions[valid_indices].numpy()
                                valid_entropy = teacher_entropy_cpu[valid_indices].numpy()
                                valid_adv = adv_cpu[valid_indices].numpy()
                                
                                # 1. Plot Teacher Entropy Scatter
                                plt.figure(figsize=(10, 6))
                                plt.scatter(valid_positions, valid_entropy, alpha=0.05, s=1)
                                plt.title(f"Teacher Entropy vs Position (Step {self.global_steps})")
                                plt.xlabel("Position")
                                plt.ylabel("Teacher Entropy")
                                plt.tight_layout()
                                entropy_plot = swanlab.Image(plt, caption=f"Teacher Entropy vs Position (Step {self.global_steps})")
                                plt.close()
                                
                                # 2. Plot Advantage Scatter
                                plt.figure(figsize=(10, 6))
                                plt.scatter(valid_positions, valid_adv, alpha=0.05, s=1)
                                plt.title(f"Advantage vs Position (Step {self.global_steps})")
                                plt.xlabel("Position")
                                plt.ylabel("Advantage")
                                plt.tight_layout()
                                adv_plot = swanlab.Image(plt, caption=f"Advantage vs Position (Step {self.global_steps})")
                                plt.close()

                                # Compute Average per Position
                                # Need to handle masking correctly.
                                # Use float tensor for mask to sum counts
                                mask_float = mask_cpu.float()
                                
                                # Sum values per position
                                sum_entropy = (teacher_entropy_cpu * mask_float).sum(dim=0)
                                sum_adv = (adv_cpu * mask_float).sum(dim=0)
                                count_per_pos = mask_float.sum(dim=0)
                                
                                # Avoid division by zero
                                valid_pos_mask = count_per_pos > 0
                                avg_entropy = torch.zeros_like(sum_entropy)
                                avg_adv = torch.zeros_like(sum_adv)
                                
                                avg_entropy[valid_pos_mask] = sum_entropy[valid_pos_mask] / count_per_pos[valid_pos_mask]
                                avg_adv[valid_pos_mask] = sum_adv[valid_pos_mask] / count_per_pos[valid_pos_mask]

                                # --- New Split Advantage Plots ---
                                avg_adv_inter = None
                                avg_adv_only_stu = None
                                overlap_mask_cpu = None
                                
                                if "overlap_mask" in batch.batch.keys():
                                    overlap_mask_cpu = batch.batch["overlap_mask"].detach().cpu()

                                if overlap_mask_cpu is not None and adv_cpu.dim() == 3:
                                    # Calculate Avg Advantage per Position for Intersection
                                    # overlap_mask_cpu: (BS, SeqLen, K)
                                    # adv_cpu: (BS, SeqLen, K)
                                    # mask_cpu: (BS, SeqLen)
                                    
                                    # Expand mask_cpu to K
                                    mask_cpu_k = mask_cpu.unsqueeze(-1).expand_as(overlap_mask_cpu)
                                    
                                    # Intersection
                                    mask_inter = (overlap_mask_cpu > 0.5) & mask_cpu_k
                                    
                                    # We sum over Batch AND K for each position
                                    sum_adv_inter = (adv_cpu * mask_inter.float()).sum(dim=(0, 2))
                                    count_inter = mask_inter.float().sum(dim=(0, 2))
                                    
                                    avg_adv_inter = torch.zeros(seq_len)
                                    valid_inter = count_inter > 0
                                    avg_adv_inter[valid_inter] = sum_adv_inter[valid_inter] / count_inter[valid_inter]
                                    
                                    # Only Stu
                                    mask_only_stu = (overlap_mask_cpu < 0.5) & mask_cpu_k
                                    
                                    sum_adv_only_stu = (adv_cpu * mask_only_stu.float()).sum(dim=(0, 2))
                                    count_only_stu = mask_only_stu.float().sum(dim=(0, 2))
                                    
                                    avg_adv_only_stu = torch.zeros(seq_len)
                                    valid_only_stu = count_only_stu > 0
                                    avg_adv_only_stu[valid_only_stu] = sum_adv_only_stu[valid_only_stu] / count_only_stu[valid_only_stu]
                                
                                # Convert to numpy for plotting
                                # We only plot positions that have at least one valid token
                                # Find the max position index that has valid data
                                if valid_pos_mask.any():
                                    max_valid_pos = torch.where(valid_pos_mask)[0].max().item()
                                    plot_positions = torch.arange(max_valid_pos + 1).numpy()
                                    plot_avg_entropy = avg_entropy[:max_valid_pos + 1].numpy()
                                    plot_avg_adv = avg_adv[:max_valid_pos + 1].numpy()
                                    
                                    plot_avg_adv_inter = avg_adv_inter[:max_valid_pos + 1].numpy() if avg_adv_inter is not None else None
                                    plot_avg_adv_only_stu = avg_adv_only_stu[:max_valid_pos + 1].numpy() if avg_adv_only_stu is not None else None
                                else:
                                    plot_positions = np.array([])
                                    plot_avg_entropy = np.array([])
                                    plot_avg_adv = np.array([])
                                    plot_avg_adv_inter = None
                                    plot_avg_adv_only_stu = None

                                # 3. Plot Average Teacher Entropy Line
                                plt.figure(figsize=(10, 6))
                                plt.plot(plot_positions, plot_avg_entropy)
                                plt.title(f"Avg Teacher Entropy vs Position (Step {self.global_steps})")
                                plt.xlabel("Position")
                                plt.ylabel("Avg Teacher Entropy")
                                plt.grid(True)
                                plt.tight_layout()
                                avg_entropy_plot = swanlab.Image(plt, caption=f"Avg Teacher Entropy vs Position (Step {self.global_steps})")
                                plt.close()

                                # 4. Plot Average Advantage Line
                                plt.figure(figsize=(10, 6))
                                plt.plot(plot_positions, plot_avg_adv, label="Total")
                                if plot_avg_adv_inter is not None:
                                    plt.plot(plot_positions, plot_avg_adv_inter, label="Intersection")
                                if plot_avg_adv_only_stu is not None:
                                    plt.plot(plot_positions, plot_avg_adv_only_stu, label="Only Stu")
                                    
                                plt.title(f"Avg Advantage vs Position (Step {self.global_steps})")
                                plt.xlabel("Position")
                                plt.ylabel("Avg Advantage")
                                plt.legend()
                                plt.grid(True)
                                plt.tight_layout()
                                avg_adv_plot = swanlab.Image(plt, caption=f"Avg Advantage vs Position (Step {self.global_steps})")
                                plt.close()
                                
                                # Log to SwanLab
                                swanlab.log({
                                    "viz/teacher_entropy_scatter": entropy_plot,
                                    "viz/advantage_scatter": adv_plot,
                                    "viz/avg_teacher_entropy_line": avg_entropy_plot,
                                    "viz/avg_advantage_line": avg_adv_plot
                                }, step=self.global_steps)
                                
                                print(f"Logged 4 plots to SwanLab at step {self.global_steps}.")
                                
                                # Free memory
                                del teacher_entropy_cpu, adv_cpu, mask_cpu, mask_float
                                del valid_positions, valid_entropy, valid_adv, positions
                                del sum_entropy, sum_adv, count_per_pos, avg_entropy, avg_adv
                                del plot_positions, plot_avg_entropy, plot_avg_adv
                                del entropy_plot, adv_plot, avg_entropy_plot, avg_adv_plot
                            else:
                                print("teacher_entropy not found in batch. Skipping plot.")
                                
                        except Exception as e:
                            print(f"Error plotting/logging: {e}")
                            import traceback
                            traceback.print_exc()

                    # Pop unused keys to save memory before PPO update
                    keys_to_pop = [
                        "teacher_on_student_log_probs",
                        "teacher_top_k_ids",
                        "teacher_top_k_log_probs",
                        "teacher_entropy",
                        "overlap_mask",
                        "teacher_in_student_mask",
                        "student_log_probs_on_teacher_ids",
                    ]
                    for key in keys_to_pop:
                        if key in batch.batch.keys():
                            batch.batch.pop(key)

                    # update critic
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, color="pink"):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with marked_timer("update_actor", timing_raw, color="red"):
                            batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    # Log rollout generations if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        self._log_rollout_data(batch, reward_extra_infos_dict, timing_raw, rollout_data_dir)

                # validate
                if (
                    self.val_reward_fn is not None
                    and self.config.trainer.test_freq > 0
                    and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                ):
                    with marked_timer("testing", timing_raw, color="green"):
                        val_metrics: dict = self._validate()
                        if is_last_step:
                            last_val_metrics = val_metrics
                    metrics.update(val_metrics)

                # Check if the ESI (Elastic Server Instance)/training plan is close to expiration.
                esi_close_to_expiration = should_save_ckpt_esi(
                    max_steps_duration=self.max_steps_duration,
                    redundant_time=self.config.trainer.esi_redundant_time,
                )
                # Check if the conditions for saving a checkpoint are met.
                # The conditions include a mandatory condition (1) and
                # one of the following optional conditions (2/3/4):
                # 1. The save frequency is set to a positive value.
                # 2. It's the last training step.
                # 3. The current step number is a multiple of the save frequency.
                # 4. The ESI(Elastic Server Instance)/training plan is close to expiration.
                if self.config.trainer.save_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.save_freq == 0 or esi_close_to_expiration
                ):
                    if esi_close_to_expiration:
                        print("Force saving checkpoint: ESI instance expiration approaching.")
                    with marked_timer("save_checkpoint", timing_raw, color="green"):
                        self._save_checkpoint()

                with marked_timer("stop_profile", timing_raw):
                    next_step_profile = (
                        self.global_steps + 1 in self.config.global_profiler.steps
                        if self.config.global_profiler.steps is not None
                        else False
                    )
                    self._stop_profiling(
                        curr_step_profile and not next_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                    prev_step_profile = curr_step_profile
                    curr_step_profile = next_step_profile

                steps_duration = timing_raw["step"]
                self.max_steps_duration = max(self.max_steps_duration, steps_duration)

                # training metrics
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
                # Note: mismatch metrics (KL, PPL, etc.) are collected at line 1179 after advantage computation

                # this is experimental and may be changed/removed in the future in favor of a general-purpose one
                if isinstance(self.train_dataloader.sampler, AbstractCurriculumSampler):
                    self.train_dataloader.sampler.update(batch=batch)

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1

                if (
                    hasattr(self.config.actor_rollout_ref.actor, "profiler")
                    and self.config.actor_rollout_ref.actor.profiler.tool == "torch_memory"
                ):
                    self.actor_rollout_wg.dump_memory_snapshot(
                        tag=f"post_update_step{self.global_steps}", sub_dir=f"step{self.global_steps}"
                    )

                if is_last_step:
                    print(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                # this is experimental and may be changed/removed in the future
                # in favor of a general-purpose data buffer pool
                if hasattr(self.train_dataset, "on_batch_end"):
                    # The dataset may be changed after each training batch
                    self.train_dataset.on_batch_end(batch=batch)
