# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
The main entry point to run the PPO algorithm
"""

import datetime
import json
import logging
import os
import warnings
from dataclasses import asdict
from typing import Any, Optional

import numpy as np
import psutil
import torch
import torch.distributed
import torch.distributed as dist
from codetiming import Timer
from omegaconf import DictConfig, OmegaConf, open_dict
import torch.distributions
from peft import LoraConfig, TaskType, get_peft_model
from safetensors.torch import save_file
from tensordict import TensorDict
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.api import FullStateDictConfig, ShardedStateDictConfig, StateDictType

try:
    # for torch 2.5+
    from torch.distributed.tensor import DTensor
except ImportError:
    from torch.distributed._tensor import DTensor

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.models.transformers.monkey_patch import apply_monkey_patch
from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import Dispatch, make_nd_compute_dataproto_dispatch_fn, register
from verl.utils import hf_processor, hf_tokenizer
from verl.utils.activation_offload import enable_activation_offloading
from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.device import (
    get_device_id,
    get_device_name,
    get_nccl_backend,
    get_torch_device,
    set_expandable_segments,
)
from verl.utils.flops_counter import FlopsCounter
from verl.utils.fs import copy_to_local
from verl.utils.fsdp_utils import (
    CPUOffloadPolicy,
    MixedPrecisionPolicy,
    apply_fsdp2,
    collect_lora_params,
    fsdp2_load_full_state_dict,
    fsdp_version,
    get_fsdp_wrap_policy,
    get_init_weight_context_manager,
    get_shard_placement_fn,
    init_fn,
    layered_summon_lora_params,
    load_fsdp_model_to_gpu,
    load_fsdp_optimizer,
    offload_fsdp_model_to_cpu,
    offload_fsdp_optimizer,
    replace_lora_wrapper,
)
from verl.utils.import_utils import import_external_libs
from verl.utils.memory_utils import aggressive_empty_cache
from verl.utils.model import compute_position_id_with_mask, convert_weight_keys
from verl.utils.profiler import DistProfiler, DistProfilerExtension, ProfilerConfig, log_gpu_memory_usage, simple_timer
from verl.utils.profiler.performance import reduce_timing, topk_reduce_ratio_min_max
from verl.utils.py_functional import convert_to_regular_types
from verl.utils.ray_utils import get_event_loop
from verl.workers.config import FSDPCriticConfig, FSDPEngineConfig, HFModelConfig, RolloutConfig
from verl.workers.config.optimizer import build_optimizer
from verl.workers.rollout import get_rollout_class
from verl.workers.sharding_manager.fsdp_ulysses import FSDPUlyssesShardingManager

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

device_name = get_device_name()


def create_device_mesh(world_size, fsdp_size):
    if fsdp_size < 0 or fsdp_size >= world_size:
        device_mesh = init_device_mesh(device_name, mesh_shape=(world_size,), mesh_dim_names=["fsdp"])
    else:
        device_mesh = init_device_mesh(
            device_name, mesh_shape=(world_size // fsdp_size, fsdp_size), mesh_dim_names=["ddp", "fsdp"]
        )
    return device_mesh


def get_sharding_strategy(device_mesh):
    from torch.distributed.fsdp import ShardingStrategy

    if device_mesh.ndim == 1:
        sharding_strategy = ShardingStrategy.FULL_SHARD
    elif device_mesh.ndim == 2:
        sharding_strategy = ShardingStrategy.HYBRID_SHARD
    else:
        raise NotImplementedError(f"Get device mesh ndim={device_mesh.ndim}, but only support 1 or 2")
    return sharding_strategy


def get_vl_model_vision_tower(vl_model_instance):
    """
    Util to extract Vision Tower from a VL model instance
    """
    if hasattr(vl_model_instance, "model") and hasattr(vl_model_instance.model, "visual"):
        # transformers >= 4.52.0
        return vl_model_instance.model.visual
    elif hasattr(vl_model_instance, "visual"):
        # transformers < 4.52.0
        return vl_model_instance.visual
    return None


class ActorRolloutRefWorker(Worker, DistProfilerExtension):
    """
    This worker can be instantiated as a standalone actor or a standalone rollout or a standalone reference policy
    or a hybrid engine based on the config.rollout
    """

    def __init__(self, config: DictConfig, role: str, **kwargs):
        Worker.__init__(self)

        self.config = config
        import torch.distributed

        if not torch.distributed.is_initialized():
            rank = int(os.environ.get("RANK", 0))
            world_size = int(os.environ.get("WORLD_SIZE", 1))
            torch.distributed.init_process_group(
                backend=f"cpu:gloo,{get_device_name()}:{get_nccl_backend()}",
                rank=rank,
                world_size=world_size,
                timeout=datetime.timedelta(seconds=self.config.get("nccl_timeout", 600)),
                init_method=os.environ.get("DIST_INIT_METHOD", None),
            )

        # build device mesh for FSDP
        world_size = torch.distributed.get_world_size()
        # TODO(sgm): support FSDP hybrid shard for larger model
        self.device_mesh = create_device_mesh(world_size=world_size, fsdp_size=self.config.actor.fsdp_config.fsdp_size)

        # build device mesh for Ulysses Sequence Parallel
        self.ulysses_device_mesh = None
        self.ulysses_sequence_parallel_size = self.config.actor.get("ulysses_sequence_parallel_size", 1)
        dp = world_size // self.ulysses_sequence_parallel_size
        if self.ulysses_sequence_parallel_size > 1:
            self.ulysses_device_mesh = init_device_mesh(
                device_name, mesh_shape=(dp, self.ulysses_sequence_parallel_size), mesh_dim_names=["dp", "sp"]
            )

        # create training dispatch
        if self.ulysses_device_mesh is not None:
            is_collect = self.ulysses_device_mesh["sp"].get_local_rank() == 0
            self._register_dispatch_collect_info(
                "actor", dp_rank=self.ulysses_device_mesh["dp"].get_local_rank(), is_collect=is_collect
            )
        else:
            self._register_dispatch_collect_info("actor", dp_rank=self.rank, is_collect=True)

        self.ulysses_sharding_manager = FSDPUlyssesShardingManager(self.ulysses_device_mesh)
        self._lora_rank = self.config.model.get("lora_rank", 0)
        self._is_lora = self.config.model.get("lora_adapter_path") is not None or self._lora_rank > 0

        self.role = role
        assert self.role in ["actor", "rollout", "ref", "actor_rollout", "actor_rollout_ref"]

        self._is_actor = self.role in ["actor", "actor_rollout", "actor_rollout_ref"]
        self._is_rollout = self.role in ["rollout", "actor_rollout", "actor_rollout_ref"]
        self._is_ref = self.role in ["ref", "actor_rollout_ref"]
        self.use_orig_params = self.config.actor.fsdp_config.get("use_orig_params", False)

        # TODO(haibin.lin):
        # As of now the type of config is DictConfig, if we assign config.profiler with ProfilerConfig,
        # it will actually convert the ProfilerConfig dataclass back to a DictConfig.
        # We can still use ProfilerConfig for testing purpose (tests/utils/test_nvtx_profile.py)
        # as they provides DictConfig-like interface
        # The benefit of creating the dataclass config is to perform validation during __post_init__
        if self._is_actor:
            omega_profiler_config = config.actor.get("profiler", {})
        elif self._is_rollout:
            # NOTE: In colocation mode, rollout config may not take effect (follow the actor config)
            # This is for extendability in AsyncRL cases
            omega_profiler_config = config.rollout.get("profiler", {})
        elif self._is_ref:
            omega_profiler_config = config.ref.get("profiler", {})
        else:
            raise ValueError(
                f"Invalid role {self.role}, should be one of "
                "['actor', 'rollout', 'ref', 'actor_rollout', 'actor_rollout_ref']"
            )
        # omega_profiler_config is DictConfig
        # profiler_config is a ProfilerConfig dataclass
        profiler_config = omega_conf_to_dataclass(omega_profiler_config, dataclass_type=ProfilerConfig)
        if omega_profiler_config.get("tool", None) in ["npu", "nsys", "torch", "torch_memory"]:
            tool_config = omega_conf_to_dataclass(
                omega_profiler_config.get("tool_config", {}).get(omega_profiler_config.get("tool"))
            )
        else:
            tool_config = None
        DistProfilerExtension.__init__(
            self, DistProfiler(rank=self.rank, config=profiler_config, tool_config=tool_config)
        )

        self._is_offload_param = False
        self._is_offload_optimizer = False
        if self._is_actor:
            self._is_offload_param = self.config.actor.fsdp_config.get("param_offload", False)
            self._is_offload_optimizer = self.config.actor.fsdp_config.get("optimizer_offload", False)
        elif self._is_ref:
            # TODO: it seems that manual offload is slowly than FSDP offload
            self._is_offload_param = self.config.ref.fsdp_config.get("param_offload", False)

        # normalize config
        if self._is_actor:
            self.config.actor.ppo_mini_batch_size *= self.config.rollout.n
            self.config.actor.ppo_mini_batch_size //= self.device_mesh.size() // self.ulysses_sequence_parallel_size
            assert self.config.actor.ppo_mini_batch_size > 0, (
                f"ppo_mini_batch_size {self.config.actor.ppo_mini_batch_size} should be larger than 0 after "
                f"normalization"
            )
            # micro bsz
            if self.config.actor.ppo_micro_batch_size is not None:
                self.config.actor.ppo_micro_batch_size //= (
                    self.device_mesh.size() // self.ulysses_sequence_parallel_size
                )
                self.config.actor.ppo_micro_batch_size_per_gpu = self.config.actor.ppo_micro_batch_size

            if self.config.actor.ppo_micro_batch_size_per_gpu is not None:
                assert self.config.actor.ppo_mini_batch_size % self.config.actor.ppo_micro_batch_size_per_gpu == 0, (
                    f"normalized ppo_mini_batch_size {self.config.actor.ppo_mini_batch_size} should be divisible by "
                    f"ppo_micro_batch_size_per_gpu {self.config.actor.ppo_micro_batch_size_per_gpu}"
                )
                assert self.config.actor.ppo_mini_batch_size // self.config.actor.ppo_micro_batch_size_per_gpu > 0, (
                    f"normalized ppo_mini_batch_size {self.config.actor.ppo_mini_batch_size} should be larger than "
                    f"ppo_micro_batch_size_per_gpu {self.config.actor.ppo_micro_batch_size_per_gpu}"
                )

        # normalize rollout config
        if self._is_rollout and self.config.rollout.log_prob_micro_batch_size is not None:
            self.config.rollout.log_prob_micro_batch_size //= (
                self.device_mesh.size() // self.ulysses_sequence_parallel_size
            )
            self.config.rollout.log_prob_micro_batch_size_per_gpu = self.config.rollout.log_prob_micro_batch_size
        # normalize ref config
        if self._is_ref and self.config.ref.log_prob_micro_batch_size is not None:
            self.config.ref.log_prob_micro_batch_size //= self.device_mesh.size() // self.ulysses_sequence_parallel_size
            self.config.ref.log_prob_micro_batch_size_per_gpu = self.config.ref.log_prob_micro_batch_size

    def _build_model_optimizer(
        self,
        model_path,
        fsdp_config: FSDPEngineConfig,
        optim_config,
        override_model_config,
        use_remove_padding=False,
        use_fused_kernels=False,
        enable_gradient_checkpointing=False,
        trust_remote_code=False,
        use_liger=False,
        role="actor",
        enable_activation_offload=False,
    ):
        from torch.distributed.fsdp import CPUOffload, MixedPrecision
        from transformers import (
            AutoConfig,
            AutoModel,
            AutoModelForCausalLM,
            AutoModelForImageTextToText,
            AutoModelForVision2Seq,
        )

        from verl.utils.model import get_generation_config, print_model_size, update_model_config
        from verl.utils.torch_dtypes import PrecisionType

        assert role in ["actor", "ref"]

        log_gpu_memory_usage(f"Before init {role} from HF AutoModel", logger=logger)
        local_path = model_path

        # note that we have to create model in fp32. Otherwise, the optimizer is in bf16, which is incorrect
        # TODO(zhangchi.usc1992): 1. support create from random initialized model. 2. Support init with FSDP directly
        self.tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        self.processor = hf_processor(local_path, trust_remote_code=trust_remote_code)

        if self.config.model.get("custom_chat_template", None) is not None:
            if self.processor is not None:
                self.processor.chat_template = self.config.model.custom_chat_template
            else:
                self.tokenizer.chat_template = self.config.model.custom_chat_template

        torch_dtype = fsdp_config.get("model_dtype", None)
        if torch_dtype is None:
            torch_dtype = torch.float32 if self._is_actor else torch.bfloat16
        else:
            torch_dtype = PrecisionType.to_dtype(torch_dtype)

        # override model kwargs
        attn_implementation = override_model_config.get("attn_implementation", "flash_attention_2")
        actor_model_config = AutoConfig.from_pretrained(
            local_path, trust_remote_code=trust_remote_code, attn_implementation=attn_implementation
        )
        # TODO: VL models use VisionAttention, which directly uses flash_attention in transformers>=4.53
        # which will be patched by _ulysses_flash_attention_forward, but errorly misses position_ids
        # Maybe support Ulysses in VisionAttention in the future and remove this patch
        if self.ulysses_sequence_parallel_size > 1 and hasattr(actor_model_config, "vision_config"):
            actor_model_config.vision_config._attn_implementation = "eager"

        # patch for kimi-vl
        if getattr(actor_model_config, "model_type", None) == "kimi_vl":
            actor_model_config.text_config.topk_method = "greedy"

        self.generation_config = get_generation_config(local_path, trust_remote_code=trust_remote_code)

        override_config_kwargs = {
            "bos_token_id": self.tokenizer.bos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        override_config_kwargs.update(override_model_config)
        update_model_config(actor_model_config, override_config_kwargs=override_config_kwargs)
        if self.rank == 0:
            print(f"Model config after override: {actor_model_config}")

        # NOTE(fix me): tie_word_embedding causes meta_tensor init to hang
        init_context = get_init_weight_context_manager(
            use_meta_tensor=not actor_model_config.tie_word_embeddings, mesh=self.device_mesh
        )

        with init_context(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            has_remote_code = hasattr(actor_model_config, "auto_map") and any(
                actor_model_config.architectures[0] in val for val in actor_model_config.auto_map.values()
            )
            if has_remote_code:
                auto_class = next(
                    k for k, v in actor_model_config.auto_map.items() if actor_model_config.architectures[0] in v
                )
                match auto_class:
                    case "AutoModelForVision2Seq":
                        actor_module_class = AutoModelForVision2Seq
                    case "AutoModelForCausalLM":
                        actor_module_class = AutoModelForCausalLM
                    case "AutoModelForImageTextToText":
                        actor_module_class = AutoModelForImageTextToText
                    case _:
                        actor_module_class = AutoModel
            else:
                if type(actor_model_config) in AutoModelForVision2Seq._model_mapping.keys():
                    actor_module_class = AutoModelForVision2Seq
                elif type(actor_model_config) in AutoModelForCausalLM._model_mapping.keys():
                    actor_module_class = AutoModelForCausalLM
                elif type(actor_model_config) in AutoModelForImageTextToText._model_mapping.keys():
                    actor_module_class = AutoModelForImageTextToText
                else:
                    actor_module_class = AutoModel

            actor_module = actor_module_class.from_pretrained(
                pretrained_model_name_or_path=local_path,
                torch_dtype=torch_dtype,
                config=actor_model_config,
                trust_remote_code=trust_remote_code,
                attn_implementation=attn_implementation,
            )

            # Apply Liger kernel to the model if use_liger is set to True
            if use_liger:
                from liger_kernel.transformers.monkey_patch import _apply_liger_kernel_to_instance

                _apply_liger_kernel_to_instance(model=actor_module)

            fused_kernel_options = self.config.model.get("fused_kernel_options", None)
            fused_kernels_backend = (
                fused_kernel_options.get("impl_backend", None) if fused_kernel_options is not None else None
            )

            apply_monkey_patch(
                model=actor_module,
                use_remove_padding=use_remove_padding,
                ulysses_sp_size=self.ulysses_sequence_parallel_size,
                use_fused_kernels=use_fused_kernels,
                fused_kernels_backend=fused_kernels_backend,
            )

            # some parameters may not in torch_dtype. TODO(zhangchi.usc1992) remove this after we switch to fsdp2
            actor_module.to(torch_dtype)

            if enable_gradient_checkpointing:
                actor_module.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

        if self._is_lora:
            print("Applying LoRA to actor module")
            actor_module.enable_input_require_grads()

            lora_adapter_path = self.config.model.get("lora_adapter_path")
            if lora_adapter_path is not None:
                from peft import PeftModel

                print(f"Loading pre-trained LoRA adapter to {role} from: {lora_adapter_path}")

                # Copy adapter to local if needed
                local_adapter_path = copy_to_local(lora_adapter_path, use_shm=self.config.model.get("use_shm", False))

                actor_module = PeftModel.from_pretrained(actor_module, local_adapter_path, is_trainable=True)
                peft_config = actor_module.peft_config["default"]
                # Ensure task_type is TaskType enum, not string
                if isinstance(peft_config.task_type, str):
                    peft_config.task_type = TaskType.CAUSAL_LM

            else:
                # Convert config to regular Python types before creating PEFT model
                lora_config = {
                    "task_type": TaskType.CAUSAL_LM,
                    "r": self.config.model.lora_rank,
                    "lora_alpha": self.config.model.lora_alpha,
                    "target_modules": convert_to_regular_types(self.config.model.target_modules),
                    "exclude_modules": convert_to_regular_types(self.config.model.exclude_modules),
                    "bias": "none",
                }
                actor_module = get_peft_model(actor_module, LoraConfig(**lora_config))

        self.use_orig_params = fsdp_config.get("use_orig_params", False)
        if self.config.actor.get("freeze_vision_tower", False):
            vision_tower = get_vl_model_vision_tower(actor_module)
            if vision_tower is not None:
                vision_tower.requires_grad_(False)
                self.use_orig_params = True
                if self.rank == 0:
                    print("[actor model] Vision tower is set to not trainable.")
            else:
                if self.rank == 0:
                    print("[actor model] No vision tower found.")

        torch.distributed.barrier()

        if self.rank == 0:
            print_model_size(actor_module)

        log_gpu_memory_usage(f"After init {role} from HF AutoModel", logger=logger)

        # We wrap FSDP for rollout as well
        mixed_precision_config = fsdp_config.get("mixed_precision", None)
        if mixed_precision_config is not None:
            param_dtype = PrecisionType.to_dtype(mixed_precision_config.get("param_dtype", "bf16"))
            reduce_dtype = PrecisionType.to_dtype(mixed_precision_config.get("reduce_dtype", "fp32"))
            buffer_dtype = PrecisionType.to_dtype(mixed_precision_config.get("buffer_dtype", "fp32"))
        else:
            param_dtype = torch.bfloat16
            reduce_dtype = torch.float32
            buffer_dtype = torch.float32

        mixed_precision = MixedPrecision(param_dtype=param_dtype, reduce_dtype=reduce_dtype, buffer_dtype=buffer_dtype)

        auto_wrap_policy = get_fsdp_wrap_policy(
            module=actor_module,
            config=fsdp_config.get("wrap_policy", None),
            is_lora=self._is_lora,
        )

        if self._is_rollout and self.config.rollout.name == "hf":
            # TODO(zhangchi.usc1992, shengguangming) fix me. Current, auto_wrap_policy causes HFRollout to hang in Gemma
            auto_wrap_policy = None

        if self.rank == 0:
            print(f"wrap_policy: {auto_wrap_policy}")

        fsdp_mesh = self.device_mesh
        sharding_strategy = get_sharding_strategy(fsdp_mesh)

        # TODO: add transformer policy
        # We force reference policy to use CPUOffload to save memory.
        # We force turn off CPUOffload for actor because it causes incorrect results when using grad accumulation
        cpu_offload = None if role == "actor" else CPUOffload(offload_params=True)
        fsdp_strategy = self.config.actor.strategy
        if fsdp_strategy == "fsdp":
            actor_module_fsdp = FSDP(
                actor_module,
                cpu_offload=cpu_offload,
                param_init_fn=init_fn,
                auto_wrap_policy=auto_wrap_policy,
                device_id=get_device_id(),
                sharding_strategy=sharding_strategy,  # zero3
                mixed_precision=mixed_precision,
                sync_module_states=True,
                device_mesh=self.device_mesh,
                use_orig_params=self.use_orig_params,
                forward_prefetch=fsdp_config.get("forward_prefetch", False),
            )
        elif fsdp_strategy == "fsdp2":
            assert CPUOffloadPolicy is not None, "PyTorch version >= 2.4 is required for using fully_shard API (FSDP2)"
            mp_policy = MixedPrecisionPolicy(
                param_dtype=param_dtype, reduce_dtype=reduce_dtype, cast_forward_inputs=True
            )
            if role == "actor" and fsdp_config.offload_policy:
                cpu_offload = CPUOffloadPolicy(pin_memory=True)
                self._is_offload_param = False
                self._is_offload_optimizer = False
            else:
                cpu_offload = None if role == "actor" else CPUOffloadPolicy(pin_memory=True)

            fsdp_kwargs = {
                "mesh": fsdp_mesh,
                "mp_policy": mp_policy,
                "offload_policy": cpu_offload,
                "reshard_after_forward": fsdp_config.reshard_after_forward,
                "shard_placement_fn": get_shard_placement_fn(fsdp_size=self.device_mesh.shape[-1]),
            }
            full_state = actor_module.state_dict()
            apply_fsdp2(actor_module, fsdp_kwargs, fsdp_config)
            fsdp2_load_full_state_dict(actor_module, full_state, fsdp_mesh, cpu_offload)
            actor_module_fsdp = actor_module
        else:
            raise NotImplementedError(f"not implement {fsdp_strategy}")

        if enable_activation_offload:
            enable_activation_offloading(actor_module_fsdp, fsdp_strategy, enable_gradient_checkpointing)

        log_gpu_memory_usage(f"After {role} FSDP init", logger=logger)

        # TODO: add more optimizer args into config
        if role == "actor" and optim_config is not None:
            from verl.utils.torch_functional import get_constant_schedule_with_warmup, get_cosine_schedule_with_warmup

            actor_optimizer = build_optimizer(actor_module_fsdp.parameters(), optim_config)

            total_steps = optim_config.get("total_training_steps", 0)
            num_warmup_steps = int(optim_config.get("lr_warmup_steps", -1))
            lr_scheduler_type = optim_config.get("lr_scheduler_type", "constant")
            min_lr_ratio = optim_config.get("min_lr_ratio", 0.0)
            num_cycles = optim_config.get("num_cycles", 0.5)
            if num_warmup_steps < 0:
                num_warmup_steps_ratio = optim_config.get("lr_warmup_steps_ratio", 0.0)
                num_warmup_steps = int(num_warmup_steps_ratio * total_steps)

            if self.rank == 0:
                print(f"Total steps: {total_steps}, num_warmup_steps: {num_warmup_steps}")

            if lr_scheduler_type == "constant":
                actor_lr_scheduler = get_constant_schedule_with_warmup(
                    optimizer=actor_optimizer, num_warmup_steps=num_warmup_steps
                )
            elif lr_scheduler_type == "cosine":
                actor_lr_scheduler = get_cosine_schedule_with_warmup(
                    optimizer=actor_optimizer,
                    num_warmup_steps=num_warmup_steps,
                    num_training_steps=total_steps,
                    min_lr_ratio=min_lr_ratio,
                    num_cycles=num_cycles,
                )
            else:
                raise NotImplementedError(f"LR scheduler type {lr_scheduler_type} is not supported")

            log_gpu_memory_usage(f"After {role} optimizer init", logger=logger)
        else:
            actor_optimizer = None
            actor_lr_scheduler = None

        return actor_module_fsdp, actor_optimizer, actor_lr_scheduler, actor_model_config

    def _build_rollout(self, trust_remote_code=False):
        from torch.distributed.device_mesh import init_device_mesh

        # 1. parse rollout and huggingface model config
        rollout_config: RolloutConfig = omega_conf_to_dataclass(self.config.rollout)
        model_config: HFModelConfig = omega_conf_to_dataclass(self.config.model, dataclass_type=HFModelConfig)
        self.model_config = model_config

        # 2. build rollout device mesh
        infer_tp = self.config.rollout.tensor_model_parallel_size * self.config.rollout.data_parallel_size
        infer_pp = self.config.rollout.pipeline_model_parallel_size
        infer_world_size = infer_tp * infer_pp
        dp = self.world_size // infer_world_size
        assert self.world_size % infer_world_size == 0, (
            f"rollout world_size: {self.world_size} is not divisible by infer_world_size: {infer_world_size}"
        )
        rollout_device_mesh = init_device_mesh(
            device_name, mesh_shape=(dp, infer_tp, infer_pp), mesh_dim_names=["dp", "infer_tp", "infer_pp"]
        )
        rollout_name = self.config.rollout.name

        if rollout_name == "hf":
            self._register_dispatch_collect_info("rollout", dp_rank=self.rank, is_collect=True)
        else:
            is_collect = (
                rollout_device_mesh["infer_tp"].get_local_rank() == 0
                and rollout_device_mesh["infer_pp"].get_local_rank() == 0
            )
            self._register_dispatch_collect_info(
                "rollout", dp_rank=rollout_device_mesh["dp"].get_local_rank(), is_collect=is_collect
            )

        # 3. init trainer and rollout random states
        self.torch_random_states = get_torch_device().get_rng_state()
        gen_dp_rank = rollout_device_mesh["dp"].get_local_rank()
        get_torch_device().manual_seed(gen_dp_rank + 1000)  # make sure all tp ranks have the same random states
        self.gen_random_states = get_torch_device().get_rng_state()
        get_torch_device().set_rng_state(self.torch_random_states)

        # 4. build rollout model
        log_gpu_memory_usage(f"Before building {self.config.rollout.name} rollout", logger=logger)
        self.rollout = get_rollout_class(rollout_config.name, rollout_config.mode)(
            config=rollout_config, model_config=model_config, device_mesh=rollout_device_mesh
        )
        log_gpu_memory_usage(f"After building {self.config.rollout.name} rollout", logger=logger)

        # Full params
        if torch.distributed.get_world_size() == 1 and fsdp_version(self.actor_module_fsdp) == 1:
            FSDP.set_state_dict_type(
                self.actor_module_fsdp,
                state_dict_type=StateDictType.FULL_STATE_DICT,
                state_dict_config=FullStateDictConfig(),
            )
        elif fsdp_version(self.actor_module_fsdp) == 1:
            FSDP.set_state_dict_type(
                self.actor_module_fsdp,
                state_dict_type=StateDictType.SHARDED_STATE_DICT,
                state_dict_config=ShardedStateDictConfig(),
            )

        # used for LoRA
        self.base_sync_done: bool = "dummy" not in self.config.rollout.load_format
        self.layered_summon = self.config.rollout.get("layered_summon", False)

        # 5. switch to trainer mode
        # NOTE: It's critical that hybrid engine in trainer mode initially to load checkpoint.
        # For sync mode, we directly switch to trainer mode here.
        # For async mode, we can't call run_until_complete here, so we will switch to trainer mode in AgentLoopManager.
        if rollout_config.mode == "sync" and self._is_actor:
            loop = get_event_loop()
            loop.run_until_complete(self.trainer_mode())

    async def rollout_mode(self):
        """Context switch hybridengine to rollout mode."""
        aggressive_empty_cache(force_sync=True)

        log_gpu_memory_usage("Before load_fsdp_model_to_gpu", logger=logger)
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)
        log_gpu_memory_usage("After load_fsdp_model_to_gpu", logger=logger)

        peft_config = None
        peft_model = getattr(self.actor_module_fsdp, "_fsdp_wrapped_module", self.actor_module_fsdp)
        if hasattr(peft_model, "peft_config"):  # LoRA
            peft_config = peft_model.peft_config.get("default", None)
            params = collect_lora_params(
                module=self.actor_module_fsdp,
                layered_summon=self.config.rollout.get("layered_summon", False),
                base_sync_done=self.base_sync_done,
            )
            if not self.base_sync_done:
                params = {replace_lora_wrapper(k, peft_config): v for k, v in params.items()}
        else:
            params = self.actor_module_fsdp.state_dict()

        params = convert_weight_keys(
            params, getattr(self.actor_module_fsdp, "_fsdp_wrapped_module", self.actor_module_fsdp)
        )

        # Special handling for LoRA with sleep_level=2:
        # When sleep_level=2, base model weights are destroyed during each sleep cycle.
        # separately collect and update LoRA weights and base model weights through their respective interfaces.
        # Here: params contains LoRA weights, base_model_params contains base model weights.
        if peft_config is not None and getattr(self.rollout, "sleep_level", None) == 2:
            base_model_params = collect_lora_params(
                module=self.actor_module_fsdp,
                layered_summon=self.layered_summon,
                base_sync_done=False,
            )
            base_model_params = {replace_lora_wrapper(k, peft_config): v for k, v in base_model_params.items()}
            base_model_params = convert_weight_keys(
                base_model_params, getattr(self.actor_module_fsdp, "_fsdp_wrapped_module", self.actor_module_fsdp)
            )

        log_gpu_memory_usage("Before offload_fsdp_model_to_cpu", logger=logger)
        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)
        log_gpu_memory_usage("After offload_fsdp_model_to_cpu", logger=logger)

        set_expandable_segments(False)

        if peft_config is not None and self.base_sync_done:
            per_tensor_param = params.items() if isinstance(params, dict) else params  # Fixed: handle dict case
        else:
            device = get_device_id()  # used when fsdp2 set cpu_offload_policy
            per_tensor_param = (
                (name, param.to(device, non_blocking=True).full_tensor() if isinstance(param, DTensor) else param)
                for name, param in params.items()
            )

        if self.config.rollout.free_cache_engine:
            await self.rollout.resume(tags=["weights"])
        log_gpu_memory_usage("After resume weights", logger=logger)

        if peft_config is not None and getattr(self.rollout, "sleep_level", None) == 2:
            per_tensor_base_params = (
                (name, param.to(device, non_blocking=True).full_tensor() if isinstance(param, DTensor) else param)
                for name, param in base_model_params.items()
            )
            await self.rollout.update_weights(per_tensor_base_params, base_sync_done=False)
            del base_model_params, per_tensor_base_params

        await self.rollout.update_weights(per_tensor_param, peft_config=peft_config, base_sync_done=self.base_sync_done)
        log_gpu_memory_usage("After update_weights", logger=logger)
        del params, per_tensor_param
        aggressive_empty_cache(force_sync=True)
        if self.config.rollout.free_cache_engine:
            await self.rollout.resume(tags=["kv_cache"])
        log_gpu_memory_usage("After resume kv_cache", logger=logger)

        self.base_sync_done = True
        # important: need to manually set the random states of each tp to be identical.
        self.torch_random_states = get_torch_device().get_rng_state()
        get_torch_device().set_rng_state(self.gen_random_states)

    async def trainer_mode(self):
        """Context switch hybridengine to trainer mode."""
        if self.config.rollout.free_cache_engine:
            log_gpu_memory_usage("Before rollout offload", logger=logger)
            await self.rollout.release()
            log_gpu_memory_usage("After rollout offload", logger=logger)

        self.actor_module_fsdp.train()

        # add empty cache after each compute
        aggressive_empty_cache(force_sync=True)

        set_expandable_segments(True)

        # restore random states
        self.gen_random_states = get_torch_device().get_rng_state()
        get_torch_device().set_rng_state(self.torch_random_states)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        from verl.workers.actor import DataParallelPPOActor

        # This is used to import external_lib into the huggingface systems
        import_external_libs(self.config.model.get("external_lib", None))

        override_model_config = OmegaConf.to_container(OmegaConf.create(self.config.model.get("override_config", {})))
        use_remove_padding = self.config.model.get("use_remove_padding", False)
        use_shm = self.config.model.get("use_shm", False)
        use_fused_kernels = self.config.model.get("use_fused_kernels", False)

        if self._is_actor or self._is_rollout:
            # we need the model for actor and rollout
            if self._is_actor:
                optim_config = self.config.actor.optim
                fsdp_config = omega_conf_to_dataclass(self.config.actor.fsdp_config)
            else:
                optim_config = None
                fsdp_config = FSDPEngineConfig()

            local_path = copy_to_local(self.config.model.path, use_shm=use_shm)
            (
                self.actor_module_fsdp,
                self.actor_optimizer,
                self.actor_lr_scheduler,
                self.actor_model_config,
            ) = self._build_model_optimizer(
                model_path=local_path,
                fsdp_config=fsdp_config,
                optim_config=optim_config,
                override_model_config=override_model_config,
                use_remove_padding=use_remove_padding,
                use_fused_kernels=use_fused_kernels,
                enable_gradient_checkpointing=self.config.model.get("enable_gradient_checkpointing", False),
                trust_remote_code=self.config.model.get("trust_remote_code", False),
                use_liger=self.config.model.get("use_liger", False),
                role="actor",
                enable_activation_offload=self.config.model.get("enable_activation_offload", False),
            )

            # get the original unwrapped module
            if fsdp_version(self.actor_module_fsdp) == 1:
                self.actor_module = self.actor_module_fsdp._fsdp_wrapped_module

            if self._is_offload_param:
                offload_fsdp_model_to_cpu(self.actor_module_fsdp)
                log_gpu_memory_usage("After offload actor model during init", logger=logger)

            if self._is_offload_optimizer:
                offload_fsdp_optimizer(optimizer=self.actor_optimizer)
                log_gpu_memory_usage("After offload actor optimizer during init", logger=logger)

        if self._is_actor:
            actor_cfg = omega_conf_to_dataclass(self.config.actor)
            self.actor = DataParallelPPOActor(
                config=actor_cfg, actor_module=self.actor_module_fsdp, actor_optimizer=self.actor_optimizer
            )

        if self._is_rollout:
            self._build_rollout(trust_remote_code=self.config.model.get("trust_remote_code", False))

        if self._is_ref:
            ref_model_path = self.config.model.path
            ref_model = self.config.ref.get("model", None)
            if ref_model is not None:
                ref_model_path = ref_model.get("path", self.config.model.path)

            if self.rank == 0:
                print("reference model:", ref_model_path)
            local_path = copy_to_local(ref_model_path, use_shm=use_shm)
            self.ref_module_fsdp = self._build_model_optimizer(
                model_path=local_path,
                fsdp_config=omega_conf_to_dataclass(self.config.ref.fsdp_config),
                optim_config=None,
                override_model_config=override_model_config,
                use_remove_padding=use_remove_padding,
                use_fused_kernels=use_fused_kernels,
                trust_remote_code=self.config.model.get("trust_remote_code", False),
                use_liger=self.config.model.get("use_liger", False),
                role="ref",
            )[0]
            OmegaConf.set_struct(self.config.ref, True)
            with open_dict(self.config.ref):
                self.config.ref.use_remove_padding = use_remove_padding
                self.config.ref.use_fused_kernels = use_fused_kernels
            self.ref_policy = DataParallelPPOActor(config=self.config.ref, actor_module=self.ref_module_fsdp)

        if self._is_actor:
            self.flops_counter = FlopsCounter(self.actor_model_config)
            self.checkpoint_manager = FSDPCheckpointManager(
                model=self.actor_module_fsdp,
                optimizer=self.actor.actor_optimizer,
                lr_scheduler=self.actor_lr_scheduler,
                processing_class=self.processor if self.processor is not None else self.tokenizer,
                checkpoint_config=self.config.actor.checkpoint,
            )

        if not self._is_actor and self._is_rollout:
            # If ActorRolloutRefWorker is initialized as a standalone rollout,
            # create a checkpoint manager for FSDP model to allow loading FSDP checkpoints for rollout.

            checkpoint_contents = OmegaConf.create({"load_contents": ["model"], "save_contents": []})
            self.checkpoint_manager = FSDPCheckpointManager(
                model=self.actor_module_fsdp,
                optimizer=None,
                lr_scheduler=None,
                processing_class=self.processor if self.processor is not None else self.tokenizer,
                checkpoint_config=checkpoint_contents,
            )

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    @DistProfiler.annotate(color="red", role="actor_update")
    def update_actor(self, data: DataProto):
        assert self._is_actor
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)
        if self._is_offload_optimizer:
            load_fsdp_optimizer(optimizer=self.actor_optimizer, device_id=get_device_id())

        with self.ulysses_sharding_manager:
            # Keep on GPU to avoid expensive CPU-GPU-CPU round trip
            # data = data.to("cpu")  # data will to device with each micro batch on actor.update_policy

            # perform training
            with Timer(name="update_policy", logger=None) as timer:
                metrics = self.actor.update_policy(data=data)
            delta_time = timer.last
            global_num_tokens = data.meta_info["global_token_num"]
            estimated_flops, promised_flops = self.flops_counter.estimate_flops(global_num_tokens, delta_time)
            metrics["perf/mfu/actor"] = (
                estimated_flops * self.config.actor.ppo_epochs / promised_flops / self.world_size
            )
            metrics["perf/max_memory_allocated_gb"] = get_torch_device().max_memory_allocated() / (1024**3)
            metrics["perf/max_memory_reserved_gb"] = get_torch_device().max_memory_reserved() / (1024**3)
            metrics["perf/cpu_memory_used_gb"] = psutil.virtual_memory().used / (1024**3)

            lr = self.actor_lr_scheduler.get_last_lr()[0]
            metrics["actor/lr"] = lr.item() if torch.is_tensor(lr) else lr
            self.actor_lr_scheduler.step()

            # TODO: here, we should return all metrics
            output = DataProto(meta_info={"metrics": metrics})

            # Metrics are small, can keep on CPU or GPU
            # output = output.to("cpu")

        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)
            log_gpu_memory_usage("After offload actor model during update_actor", logger=logger)
        if self._is_offload_optimizer:
            offload_fsdp_optimizer(optimizer=self.actor_optimizer)
            log_gpu_memory_usage("After offload actor optimizer during update_actor", logger=logger)

        return output

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="rollout"))
    @DistProfiler.annotate(color="red", role="rollout_generate")
    def generate_sequences(self, prompts: DataProto):
        # Support all hardwares
        assert self._is_rollout
        prompts = prompts.to(get_device_id())

        meta_info = {
            "eos_token_id": self.generation_config.eos_token_id
            if self.generation_config is not None
            else self.tokenizer.eos_token_id,
            "pad_token_id": self.generation_config.pad_token_id
            if self.generation_config is not None
            else self.tokenizer.pad_token_id,
        }
        prompts.meta_info.update(meta_info)

        timing_generate = {}
        if self._is_actor:  # For rollout only, we do not switch context.
            loop = get_event_loop()
            loop.run_until_complete(self.rollout_mode())
            log_gpu_memory_usage("After switch to rollout mode", logger=logger)

        with simple_timer("generate_sequences", timing_generate):
            output = self.rollout.generate_sequences(prompts=prompts)

        if self._is_actor:
            loop.run_until_complete(self.trainer_mode())
            log_gpu_memory_usage("After switch to trainer mode", logger=logger)

        # We calculate the average timing across all ranks
        # to make sure meta_info["timing"] is the same
        timing_generate_topk_ratio, timing_generate_min, timing_generate_max = topk_reduce_ratio_min_max(
            timing_generate["generate_sequences"]
        )
        timing_generate = reduce_timing(timing_generate)
        timing_generate.update(
            {
                "generation_timing/max": timing_generate_max,
                "generation_timing/min": timing_generate_min,
                "generation_timing/topk_ratio": timing_generate_topk_ratio,
            }
        )
        output.meta_info["timing"] = timing_generate
        output = output.to("cpu")

        # clear kv cache
        get_torch_device().empty_cache()
        return output

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    @DistProfiler.annotate(color="blue", role="actor_compute_log_prob")
    def compute_log_prob(self, data: DataProto):
        # when is_lora is True, we use the actor without lora applied to calculate the log_prob
        # which is mostly used for ref log_prob calculation
        assert self._is_actor
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)

        # Support all hardwares
        from contextlib import nullcontext

        is_lora = data.meta_info.pop("is_lora", False)
        adapter_ctx = self.actor.actor_module.disable_adapter() if is_lora else nullcontext()
        # we should always recompute old_log_probs when it is HybridEngine
        data.meta_info["micro_batch_size"] = self.config.rollout.log_prob_micro_batch_size_per_gpu
        data.meta_info["max_token_len"] = self.config.rollout.log_prob_max_token_len_per_gpu
        data.meta_info["use_dynamic_bsz"] = self.config.rollout.log_prob_use_dynamic_bsz
        data.meta_info["temperature"] = self.config.rollout.temperature
        data.meta_info["top_k"] = self.config.rollout.get("log_prob_top_k", 0)
        # data.meta_info["top_p"] = 1.0
        # print("log_prob_top_k", data.meta_info["top_k"])
        # perform recompute log_prob
        with self.ulysses_sharding_manager:
            with adapter_ctx:
                output, entropys, topk_ids, topk_log_probs = self.actor.compute_log_prob(data=data, calculate_entropy=True)
            
            tensors = {"old_log_probs": output, "entropys": entropys}
            if topk_ids is not None:
                tensors["student_top_k_ids"] = topk_ids
            if topk_log_probs is not None:
                tensors["student_top_k_log_probs"] = topk_log_probs
                tensors["student_valid_counts"] = (topk_log_probs > -1e6).sum(dim=-1)
            
            output = DataProto.from_dict(
                tensors=tensors,
                meta_info={"temperature": self.config.rollout.temperature},
            )

        # Keep on GPU to avoid expensive CPU-GPU transfer for large top-k data
        # output = output.to("cpu")

        # https://pytorch.org/docs/stable/notes/fsdp.html#fsdp-notes
        # unshard the root FSDP module
        if self.world_size > 1 and fsdp_version(self.actor.actor_module) == 1:
            self.actor.actor_module._handle.reshard(True)

        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)
            log_gpu_memory_usage("After offload actor model during compute_log_prob", logger=logger)

        return output

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    @DistProfiler.annotate(color="blue", role="actor_compute_log_probs_for_ids")
    def compute_log_probs_for_ids(self, data: DataProto):
        assert self._is_actor
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)

        # Support all hardwares
        # we should always recompute old_log_probs when it is HybridEngine
        data.meta_info["micro_batch_size"] = self.config.rollout.log_prob_micro_batch_size_per_gpu
        data.meta_info["max_token_len"] = self.config.rollout.log_prob_max_token_len_per_gpu
        data.meta_info["use_dynamic_bsz"] = self.config.rollout.log_prob_use_dynamic_bsz
        data.meta_info["temperature"] = self.config.rollout.temperature
        
        with self.ulysses_sharding_manager:
            output = self.actor.compute_log_probs_for_ids(data=data)
            output = DataProto.from_dict(
                tensors={"student_log_probs_on_teacher_ids": output},
                meta_info={"temperature": self.config.rollout.temperature},
            )

        # Keep on GPU to avoid expensive CPU-GPU transfer for large top-k data
        # output = output.to("cpu")

        # https://pytorch.org/docs/stable/notes/fsdp.html#fsdp-notes
        # unshard the root FSDP module
        if self.world_size > 1 and fsdp_version(self.actor.actor_module) == 1:
            self.actor.actor_module._handle.reshard(True)

        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)
            log_gpu_memory_usage("After offload actor model during compute_log_probs_for_ids", logger=logger)

        return output

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    @DistProfiler.annotate(color="blue", role="actor_compute_distillation_reward")
    def compute_distillation_reward(self, data: DataProto):
        assert self._is_actor
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)

        # Support all hardwares
        # we should always recompute old_log_probs when it is HybridEngine
        data.meta_info["micro_batch_size"] = self.config.rollout.log_prob_micro_batch_size_per_gpu
        data.meta_info["max_token_len"] = self.config.rollout.log_prob_max_token_len_per_gpu
        data.meta_info["use_dynamic_bsz"] = self.config.rollout.log_prob_use_dynamic_bsz
        data.meta_info["temperature"] = self.config.rollout.temperature
        
        with self.ulysses_sharding_manager:
            output = self.actor.compute_distillation_reward(data=data)

        # https://pytorch.org/docs/stable/notes/fsdp.html#fsdp-notes
        # unshard the root FSDP module
        if self.world_size > 1 and fsdp_version(self.actor.actor_module) == 1:
            self.actor.actor_module._handle.reshard(True)

        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)
            log_gpu_memory_usage("After offload actor model during compute_distillation_reward", logger=logger)

        return output

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    @DistProfiler.annotate(color="olive", role="ref_compute_log_prob")
    def compute_ref_log_prob(self, data: DataProto):
        if self._is_lora:
            # if _is_lora, actor without lora applied is the ref
            data.meta_info["is_lora"] = True
            data = self.compute_log_prob(data)
            # this old_log_probs is in fact ref_log_prob
            data = DataProto.from_dict(tensors={"ref_log_prob": data.batch["old_log_probs"]})
            return data
        assert self._is_ref
        # else:
        # otherwise, the class have a standalone ref model

        micro_batch_size = self.config.ref.log_prob_micro_batch_size_per_gpu
        data.meta_info["micro_batch_size"] = micro_batch_size
        data.meta_info["temperature"] = self.config.rollout.temperature
        data.meta_info["max_token_len"] = self.config.ref.log_prob_max_token_len_per_gpu
        data.meta_info["use_dynamic_bsz"] = self.config.ref.log_prob_use_dynamic_bsz
        data.meta_info["top_k"] = 0
        with self.ulysses_sharding_manager:
            data = data.to("cpu")  # data will to device with each micro batch on ref.compute_log_prob
            output, _, _, _ = self.ref_policy.compute_log_prob(data=data, calculate_entropy=False)
            output = DataProto.from_dict(tensors={"ref_log_prob": output})

        output = output.to("cpu")

        # https://pytorch.org/docs/stable/notes/fsdp.html#fsdp-notes
        # unshard the root FSDP module
        if self.world_size > 1:
            if fsdp_version(self.ref_policy.actor_module) == 1:
                self.ref_policy.actor_module._handle.reshard(True)
            elif fsdp_version(self.ref_policy.actor_module) == 2:
                self.ref_policy.actor_module.reshard()

        return output

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_checkpoint(self, local_path, hdfs_path=None, global_step=0, max_ckpt_to_keep=None):
        from verl.utils.logger import log_with_rank

        # only support save and load ckpt for actor
        assert self._is_actor

        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)

        self.checkpoint_manager.save_checkpoint(
            local_path=local_path, hdfs_path=hdfs_path, global_step=global_step, max_ckpt_to_keep=max_ckpt_to_keep
        )
        dist.barrier()

        if self._is_lora and hasattr(getattr(self, "actor_module", self.actor_module_fsdp), "peft_config"):
            lora_save_path = os.path.join(local_path, "lora_adapter")
            peft_model = getattr(self, "actor_module", self.actor_module_fsdp)
            peft_config = {}
            if dist.get_rank() == 0:
                os.makedirs(lora_save_path, exist_ok=True)
                peft_config = asdict(peft_model.peft_config.get("default", {}))
                peft_config["task_type"] = peft_config["task_type"].value
                peft_config["peft_type"] = peft_config["peft_type"].value
                peft_config["target_modules"] = list(peft_config["target_modules"])
            try:
                if fsdp_version(self.actor_module_fsdp) > 0:
                    self.actor_module_fsdp = self.actor_module_fsdp.to(get_device_name())
                    lora_params = layered_summon_lora_params(self.actor_module_fsdp)
                    if dist.get_rank() == 0:
                        save_file(lora_params, os.path.join(lora_save_path, "adapter_model.safetensors"))
                        with open(os.path.join(lora_save_path, "adapter_config.json"), "w", encoding="utf-8") as f:
                            json.dump(peft_config, f, ensure_ascii=False, indent=4)
            except Exception as e:
                log_with_rank(
                    f"Save LoRA Adapter Error ({e})", rank=dist.get_rank(), logger=logger, log_only_rank_0=True
                )

            dist.barrier()
            log_with_rank(
                f"[rank-{self.rank}]: Saved LoRA adapter to: {lora_save_path}",
                rank=dist.get_rank(),
                logger=logger,
                log_only_rank_0=True,
            )

        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_checkpoint(self, local_path, hdfs_path=None, del_local_after_load=False):
        assert self._is_actor or (not self._is_actor and self._is_rollout), (
            f"Checkpoint loading is only supported for Actor or standalone Rollout Workers, but got "
            f"{self._is_actor} and {self._is_rollout}"
        )

        # No checkpoint to load, just offload the model and optimizer to CPU
        if local_path is None:
            if self._is_offload_param:
                offload_fsdp_model_to_cpu(self.actor_module_fsdp)
            if self._is_offload_optimizer:
                offload_fsdp_optimizer(self.actor_optimizer)
            return

        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)

        self.checkpoint_manager.load_checkpoint(
            local_path=local_path, hdfs_path=hdfs_path, del_local_after_load=del_local_after_load
        )

        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)

        if self._is_offload_optimizer:
            offload_fsdp_optimizer(self.actor_optimizer)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def start_profile(self, **kwargs) -> None:
        """Start profiling for the current rank in the current training step."""
        self.profiler.start(**kwargs)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def stop_profile(self) -> None:
        """Stop profiling for the current rank in the current training step."""
        self.profiler.stop()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def dump_memory_snapshot(self, tag: str = "manual", sub_dir: str = None) -> None:
        """Manually trigger a CUDA memory snapshot dump on all ranks."""
        # Memory snapshot is now handled by the profiler system
        # This method is kept for backward compatibility but delegates to profiler
        if hasattr(self, "profiler") and hasattr(self.profiler, "_impl"):
            try:
                # Try to use the profiler's memory snapshot functionality
                if hasattr(self.profiler._impl, "sampler"):
                    out_dir = OmegaConf.select(self.config, "actor.profiler.save_path") or "."
                    self.profiler._impl.sampler.dump_memory_snapshot(out_dir=out_dir, tag=tag, sub_dir=sub_dir)
            except Exception:
                # silently ignore if profiler doesn't support memory snapshots
                pass


class CriticWorker(Worker, DistProfilerExtension):
    def __init__(self, config: FSDPCriticConfig):
        Worker.__init__(self)
        omega_profiler_config = config.get("profiler", {})
        profiler_config = omega_conf_to_dataclass(omega_profiler_config, dataclass_type=ProfilerConfig)
        if omega_profiler_config.get("tool", None) in ["npu", "nsys", "torch", "torch_memory"]:
            tool_config = omega_conf_to_dataclass(
                omega_profiler_config.get("tool_config", {}).get(omega_profiler_config.get("tool"))
            )
        else:
            tool_config = None
        DistProfilerExtension.__init__(
            self, DistProfiler(rank=self.rank, config=profiler_config, tool_config=tool_config)
        )
        import torch.distributed

        self.config = config
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(
                backend=get_nccl_backend(),
                timeout=datetime.timedelta(seconds=self.config.get("nccl_timeout", 600)),
                init_method=os.environ.get("DIST_INIT_METHOD", None),
            )
        self.config: FSDPCriticConfig = config

        # build device mesh for Ulysses Sequence Parallel
        world_size = torch.distributed.get_world_size()
        from torch.distributed.device_mesh import init_device_mesh

        fsdp_size = self.config.model.fsdp_config.fsdp_size
        self.device_mesh = create_device_mesh(world_size=world_size, fsdp_size=fsdp_size)

        self.ulysses_device_mesh = None
        self.ulysses_sequence_parallel_size = self.config.get("ulysses_sequence_parallel_size", 1)
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1
        dp = world_size // self.ulysses_sequence_parallel_size
        if self.ulysses_sequence_parallel_size > 1:
            self.ulysses_device_mesh = init_device_mesh(
                device_name, mesh_shape=(dp, self.ulysses_sequence_parallel_size), mesh_dim_names=["dp", "sp"]
            )

        # create training dispatch
        if self.ulysses_device_mesh is not None:
            is_collect = self.ulysses_device_mesh["sp"].get_local_rank() == 0
            self._register_dispatch_collect_info(
                "critic", dp_rank=self.ulysses_device_mesh["dp"].get_local_rank(), is_collect=is_collect
            )
        else:
            self._register_dispatch_collect_info("critic", dp_rank=self.rank, is_collect=True)

        self.ulysses_sharding_manager = FSDPUlyssesShardingManager(self.ulysses_device_mesh)

        # set FSDP offload params
        self._is_offload_param = self.config.model.fsdp_config.param_offload
        self._is_offload_optimizer = self.config.model.fsdp_config.optimizer_offload

        # normalize config
        self.config.ppo_mini_batch_size *= self.config.rollout_n
        self.config.ppo_mini_batch_size //= torch.distributed.get_world_size() // self.ulysses_sequence_parallel_size
        if self.config.ppo_micro_batch_size is not None:
            self.config.ppo_micro_batch_size //= (
                torch.distributed.get_world_size() // self.ulysses_sequence_parallel_size
            )
            self.config.forward_micro_batch_size //= (
                torch.distributed.get_world_size() // self.ulysses_sequence_parallel_size
            )
            self.config.ppo_micro_batch_size_per_gpu = self.config.ppo_micro_batch_size
            self.config.forward_micro_batch_size_per_gpu = self.config.forward_micro_batch_size

        if self.config.ppo_micro_batch_size_per_gpu is not None:
            assert self.config.ppo_mini_batch_size % self.config.ppo_micro_batch_size_per_gpu == 0, (
                f"normalized ppo_mini_batch_size {self.config.ppo_mini_batch_size} should be divisible by "
                f"ppo_micro_batch_size_per_gpu {self.config.ppo_micro_batch_size_per_gpu}"
            )
            assert self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu > 0, (
                f"normalized ppo_mini_batch_size {self.config.ppo_mini_batch_size} should be larger than "
                f"ppo_micro_batch_size_per_gpu {self.config.ppo_micro_batch_size_per_gpu}"
            )
        self._is_lora = (
            self.config.model.get("lora_adapter_path") is not None or self.config.model.get("lora_rank", 0) > 0
        )
        self.use_orig_params = self.config.model.fsdp_config.get("use_orig_params", False)

    def _build_critic_model_optimizer(self, config):
        # the following line is necessary
        from torch.distributed.fsdp import MixedPrecision

        from verl.utils.model import load_valuehead_model, print_model_size
        from verl.utils.torch_dtypes import PrecisionType

        use_shm = config.model.get("use_shm", False)
        local_path = copy_to_local(config.model.path, use_shm=use_shm)
        # note that the tokenizer between actor and critic may be different. So override tokenizer info with actor info
        # using random initialized model from any architecture. May not be the same as Actor.

        tokenizer_path = copy_to_local(config.model.tokenizer_path, use_shm=use_shm)
        self.tokenizer = hf_tokenizer(tokenizer_path, trust_remote_code=config.model.get("trust_remote_code", False))
        self.processor = hf_processor(tokenizer_path, trust_remote_code=config.model.get("trust_remote_code", False))

        if self.config.model.get("custom_chat_template", None) is not None:
            if self.processor is not None:
                self.processor.chat_template = self.config.model.custom_chat_template
            else:
                self.tokenizer.chat_template = self.config.model.custom_chat_template
        override_config = OmegaConf.to_container(OmegaConf.create(self.config.model.get("override_config", {})))
        override_config_kwargs = {
            "bos_token_id": self.tokenizer.bos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        override_config_kwargs.update(override_config)
        if self.rank == 0:
            print(f"Critic overriding config {override_config_kwargs}")

        torch_dtype = self.config.model.fsdp_config.get("model_dtype", "fp32")
        torch_dtype = PrecisionType.to_dtype(torch_dtype)

        from transformers import AutoConfig

        # override model kwargs
        attn_implementation = override_config.get("attn_implementation", "flash_attention_2")
        critic_model_config = AutoConfig.from_pretrained(
            local_path,
            attn_implementation=attn_implementation,
            trust_remote_code=config.model.get("trust_remote_code", False),
        )
        # TODO: VL models use VisionAttention, which directly uses flash_attention in transformers>=4.53
        # which will be patched by _ulysses_flash_attention_forward, but errorly misses position_ids
        # Maybe support Ulysses in VisionAttention in the future and remove this patch
        if self.ulysses_sequence_parallel_size > 1 and hasattr(critic_model_config, "vision_config"):
            critic_model_config.vision_config._attn_implementation = "eager"

        critic_model_config.num_labels = 1
        # patch for kimi-vl
        if getattr(critic_model_config, "model_type", None) == "kimi_vl":
            critic_model_config.text_config.topk_method = "greedy"

        init_context = get_init_weight_context_manager(
            use_meta_tensor=not critic_model_config.tie_word_embeddings, mesh=self.device_mesh
        )

        with init_context(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            critic_model_config.classifier_dropout = 0.0
            critic_model_config.hidden_dropout = "0"
            critic_model_config.summary_dropout_prob = 0.0

            critic_module = load_valuehead_model(
                local_path,
                torch_dtype,
                critic_model_config,
                config.model.get("trust_remote_code", False),
            )

            use_remove_padding = config.model.get("use_remove_padding", False)

            apply_monkey_patch(
                model=critic_module,
                use_remove_padding=use_remove_padding,
                ulysses_sp_size=self.ulysses_sequence_parallel_size,
            )

            # some parameters may not in torch_dtype
            critic_module.to(torch_dtype)

            if config.model.get("enable_gradient_checkpointing", False):
                critic_module.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

        if self._is_lora:
            print("Applying LoRA to critic module")
            critic_module.enable_input_require_grads()

            # Check if we should load a pre-trained LoRA adapter
            lora_adapter_path = self.config.model.get("lora_adapter_path")
            if lora_adapter_path is not None:
                from peft import PeftModel

                print(f"Loading pre-trained LoRA adapter to critic from: {lora_adapter_path}")

                # Copy adapter to local if needed
                local_adapter_path = copy_to_local(lora_adapter_path, use_shm=self.config.model.get("use_shm", False))

                critic_module = PeftModel.from_pretrained(critic_module, local_adapter_path, is_trainable=True)
                peft_config = critic_module.peft_config["default"]
                # Ensure task_type is TaskType enum, not string
                if isinstance(peft_config.task_type, str):
                    peft_config.task_type = TaskType.CAUSAL_LM

            else:
                # Convert config to regular Python types before creating PEFT model
                lora_config = {
                    "task_type": TaskType.CAUSAL_LM,
                    "r": self.config.model.lora_rank,
                    "lora_alpha": self.config.model.lora_alpha,
                    "target_modules": convert_to_regular_types(self.config.model.target_modules),
                    "bias": "none",
                }
                critic_module = get_peft_model(critic_module, LoraConfig(**lora_config))

        if self.rank == 0:
            print_model_size(critic_module)

        self.critic_model_config = critic_model_config

        fsdp_config = self.config.model.fsdp_config
        mixed_precision_config = fsdp_config.get("mixed_precision", None)
        if mixed_precision_config is not None:
            param_dtype = PrecisionType.to_dtype(mixed_precision_config.get("param_dtype", "bf16"))
            reduce_dtype = PrecisionType.to_dtype(mixed_precision_config.get("reduce_dtype", "fp32"))
            buffer_dtype = PrecisionType.to_dtype(mixed_precision_config.get("buffer_dtype", "fp32"))
        else:
            param_dtype = torch.bfloat16
            reduce_dtype = torch.float32
            buffer_dtype = torch.float32

        mixed_precision = MixedPrecision(param_dtype=param_dtype, reduce_dtype=reduce_dtype, buffer_dtype=buffer_dtype)

        auto_wrap_policy = get_fsdp_wrap_policy(
            module=critic_module,
            config=self.config.model.fsdp_config.wrap_policy,
            is_lora=self._is_lora,
        )

        log_gpu_memory_usage("Before critic FSDP", logger=None)

        fsdp_mesh = self.device_mesh
        sharding_strategy = get_sharding_strategy(fsdp_mesh)

        self.use_orig_params = fsdp_config.get("use_orig_params", False)
        if self.config.model.get("freeze_vision_tower", False):
            vision_tower = get_vl_model_vision_tower(critic_module)
            if vision_tower is not None:
                vision_tower.requires_grad_(False)
                self.use_orig_params = True
                if self.rank == 0:
                    print("[critic model] Vision tower is set to not trainable.")
            else:
                if self.rank == 0:
                    print("[critic model] No vision tower found.")

        # Note: We force turn off CPUOffload for critic because it causes incorrect results when using grad accumulation
        if config.strategy == "fsdp":
            critic_module = FSDP(
                critic_module,
                param_init_fn=init_fn,
                use_orig_params=self.use_orig_params,
                auto_wrap_policy=auto_wrap_policy,
                device_id=get_device_id(),
                sharding_strategy=sharding_strategy,
                mixed_precision=mixed_precision,
                sync_module_states=True,
                forward_prefetch=self.config.model.fsdp_config.forward_prefetch,
                device_mesh=self.device_mesh,
                cpu_offload=None,
            )
        elif config.strategy == "fsdp2":
            assert CPUOffloadPolicy is not None, "PyTorch version >= 2.4 is required for using fully_shard API (FSDP2)"
            mp_policy = MixedPrecisionPolicy(
                param_dtype=param_dtype, reduce_dtype=reduce_dtype, cast_forward_inputs=True
            )
            offload_policy = None
            if fsdp_config.offload_policy:
                self._is_offload_param = False
                self._is_offload_optimizer = False
                offload_policy = CPUOffloadPolicy(pin_memory=True)

            fsdp_kwargs = {
                "mesh": fsdp_mesh,
                "mp_policy": mp_policy,
                "offload_policy": offload_policy,
                "reshard_after_forward": fsdp_config.reshard_after_forward,
                "shard_placement_fn": get_shard_placement_fn(fsdp_size=self.device_mesh.shape[-1]),
            }
            full_state = critic_module.state_dict()
            apply_fsdp2(critic_module, fsdp_kwargs, fsdp_config)
            fsdp2_load_full_state_dict(critic_module, full_state, fsdp_mesh, offload_policy)
        else:
            raise NotImplementedError(f"Unknown strategy {config.strategy}")

        if config.model.get("enable_activation_offload", False):
            enable_gradient_checkpointing = config.model.get("enable_gradient_checkpointing", False)
            enable_activation_offloading(critic_module, config.strategy, enable_gradient_checkpointing)

        log_gpu_memory_usage("After critic FSDP", logger=None)

        critic_optimizer = build_optimizer(critic_module.parameters(), config.optim)

        total_steps = config.optim.get("total_training_steps", 0)
        num_warmup_steps = int(config.optim.get("lr_warmup_steps", -1))

        lr_scheduler_type = config.optim.get("lr_scheduler_type", "constant")
        if num_warmup_steps < 0:
            num_warmup_steps_ratio = config.optim.get("lr_warmup_steps_ratio", 0.0)
            num_warmup_steps = int(num_warmup_steps_ratio * total_steps)

        if self.rank == 0:
            print(f"Total steps: {total_steps}, num_warmup_steps: {num_warmup_steps}")

        from verl.utils.torch_functional import get_constant_schedule_with_warmup, get_cosine_schedule_with_warmup

        if lr_scheduler_type == "constant":
            critic_lr_scheduler = get_constant_schedule_with_warmup(
                optimizer=critic_optimizer, num_warmup_steps=num_warmup_steps
            )
        elif lr_scheduler_type == "cosine":
            min_lr_ratio = config.optim.get("min_lr_ratio", 0.0)
            num_cycles = config.optim.get("num_cycles", 0.5)
            critic_lr_scheduler = get_cosine_schedule_with_warmup(
                optimizer=critic_optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=total_steps,
                min_lr_ratio=min_lr_ratio,
                num_cycles=num_cycles,
            )
        else:
            raise NotImplementedError(f"LR scheduler type {lr_scheduler_type} is not supported")

        return critic_module, critic_optimizer, critic_lr_scheduler

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        # This is used to import external_lib into the huggingface systems
        import_external_libs(self.config.model.get("external_lib", None))

        from verl.workers.critic import DataParallelPPOCritic

        self.critic_module, self.critic_optimizer, self.critic_lr_scheduler = self._build_critic_model_optimizer(
            self.config
        )

        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.critic_module)
            log_gpu_memory_usage("After offload critic model during init", logger=logger)
        if self._is_offload_optimizer:
            offload_fsdp_optimizer(optimizer=self.critic_optimizer)
            log_gpu_memory_usage("After offload critic optimizer during init", logger=logger)

        self.critic = DataParallelPPOCritic(
            config=self.config, critic_module=self.critic_module, critic_optimizer=self.critic_optimizer
        )

        self.flops_counter = FlopsCounter(self.critic_model_config)
        self.checkpoint_manager = FSDPCheckpointManager(
            model=self.critic_module,
            optimizer=self.critic_optimizer,
            lr_scheduler=self.critic_lr_scheduler,
            processing_class=self.processor if self.processor is not None else self.tokenizer,
            checkpoint_config=self.config.checkpoint,
        )

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="critic"))
    @DistProfiler.annotate(color="cyan")
    def compute_values(self, data: DataProto):
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.critic_module)
        micro_batch_size = self.config.forward_micro_batch_size_per_gpu
        data.meta_info["micro_batch_size"] = micro_batch_size
        data.meta_info["max_token_len"] = self.config.forward_max_token_len_per_gpu
        data.meta_info["use_dynamic_bsz"] = self.config.use_dynamic_bsz
        # perform forward computation
        with self.ulysses_sharding_manager:
            data = data.to("cpu")  # data will to device with each micro batch on critic.compute_values
            values = self.critic.compute_values(data=data)
            output = DataProto.from_dict(tensors={"values": values})

        output = output.to("cpu")
        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.critic_module)
        return output

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="critic"))
    @DistProfiler.annotate(color="pink")
    def update_critic(self, data: DataProto):
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.critic_module)
        if self._is_offload_optimizer:
            load_fsdp_optimizer(optimizer=self.critic_optimizer, device_id=get_device_id())

        # perform forward computation
        with self.ulysses_sharding_manager:
            data = data.to("cpu")  # data will to device with each micro batch on critic.update_critic
            with Timer(name="update_critic", logger=None) as timer:
                metrics = self.critic.update_critic(data=data)
            delta_time = timer.last

            global_num_tokens = data.meta_info["global_token_num"]
            estimated_flops, promised_flops = self.flops_counter.estimate_flops(global_num_tokens, delta_time)
            metrics["perf/mfu/critic"] = estimated_flops * self.config.ppo_epochs / promised_flops / self.world_size

            lr = self.critic_lr_scheduler.get_last_lr()[0]
            metrics["critic/lr"] = lr
            self.critic_lr_scheduler.step()

            output = DataProto(batch=None, meta_info={"metrics": metrics})

        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.critic_module)
        if self._is_offload_optimizer:
            offload_fsdp_optimizer(optimizer=self.critic_optimizer)

        output = output.to("cpu")
        return output

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_checkpoint(self, local_path, hdfs_path=None, global_step=0, max_ckpt_to_keep=None):
        import torch

        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.critic_module)

        self.checkpoint_manager.save_checkpoint(
            local_path=local_path, hdfs_path=hdfs_path, global_step=global_step, max_ckpt_to_keep=max_ckpt_to_keep
        )

        torch.distributed.barrier()
        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.critic_module)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_checkpoint(self, local_path, hdfs_path=None, del_local_after_load=True):
        import torch

        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.critic_module)

        self.checkpoint_manager.load_checkpoint(
            local_path=local_path, hdfs_path=hdfs_path, del_local_after_load=del_local_after_load
        )

        torch.distributed.barrier()
        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.critic_module)

        if self._is_offload_optimizer:
            offload_fsdp_optimizer(self.critic_optimizer)


# TODO(sgm): we may need to extract it to dp_reward_model.py
class RewardModelWorker(Worker, DistProfilerExtension):
    """
    Note that we only implement the reward model that is subclass of AutoModelForTokenClassification.
    """

    def __init__(self, config):
        Worker.__init__(self)

        omega_profiler_config = config.get("profiler", {})
        profiler_config = omega_conf_to_dataclass(omega_profiler_config, dataclass_type=ProfilerConfig)
        if omega_profiler_config.get("tool", None) in ["npu", "nsys", "torch", "torch_memory"]:
            tool_config = omega_conf_to_dataclass(
                omega_profiler_config.get("tool_config", {}).get(omega_profiler_config.get("tool"))
            )
        else:
            tool_config = None
        DistProfilerExtension.__init__(
            self,
            DistProfiler(rank=self.rank, config=profiler_config, tool_config=tool_config),
        )

        import torch.distributed

        self.config = config
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(
                backend=get_nccl_backend(),
                timeout=datetime.timedelta(seconds=self.config.get("nccl_timeout", 600)),
                init_method=os.environ.get("DIST_INIT_METHOD", None),
            )

        # build device mesh for Ulysses Sequence Parallel
        world_size = torch.distributed.get_world_size()
        from torch.distributed.device_mesh import init_device_mesh

        fsdp_size = self.config.model.fsdp_config.fsdp_size
        self.device_mesh = create_device_mesh(world_size=world_size, fsdp_size=fsdp_size)

        self.ulysses_device_mesh = None
        self.ulysses_sequence_parallel_size = self.config.get("ulysses_sequence_parallel_size", 1)
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1
        dp = world_size // self.ulysses_sequence_parallel_size
        if self.ulysses_sequence_parallel_size > 1:
            self.ulysses_device_mesh = init_device_mesh(
                device_name, mesh_shape=(dp, self.ulysses_sequence_parallel_size), mesh_dim_names=["dp", "sp"]
            )

        self.ulysses_sharding_manager = FSDPUlyssesShardingManager(self.ulysses_device_mesh)

        # create training dispatch
        if self.ulysses_device_mesh is not None:
            is_collect = self.ulysses_device_mesh["sp"].get_local_rank() == 0
            self._register_dispatch_collect_info(
                "reward", dp_rank=self.ulysses_device_mesh["dp"].get_local_rank(), is_collect=is_collect
            )
        else:
            self._register_dispatch_collect_info("reward", dp_rank=self.rank, is_collect=True)

        self.use_remove_padding = self.config.model.get("use_remove_padding", False)
        self.use_fused_kernels = self.config.model.get("use_fused_kernels", False)

        # normalize config
        if self.config.micro_batch_size is not None:
            self.config.micro_batch_size //= torch.distributed.get_world_size()
            self.config.micro_batch_size_per_gpu = self.config.micro_batch_size

    def _build_model(self, config):
        # the following line is necessary
        from torch.distributed.fsdp import CPUOffload
        from transformers import AutoConfig, AutoModelForCausalLM

        use_shm = config.model.get("use_shm", False)
        # download the checkpoint from hdfs
        local_path = copy_to_local(config.model.path, use_shm=use_shm)

        if self.config.model.input_tokenizer is None:
            self._do_switch_chat_template = False
        else:
            self._do_switch_chat_template = True
            input_tokenizer_local_path = copy_to_local(config.model.input_tokenizer, use_shm=use_shm)
            self.input_tokenizer = hf_tokenizer(
                input_tokenizer_local_path, trust_remote_code=config.model.get("trust_remote_code", False)
            )
            self.tokenizer = hf_tokenizer(local_path, trust_remote_code=config.model.get("trust_remote_code", False))

        trust_remote_code = config.model.get("trust_remote_code", False)
        model_config = AutoConfig.from_pretrained(local_path, trust_remote_code=trust_remote_code)

        # note that we have to create model in fp32. Otherwise, the optimizer is in bf16, which is incorrect
        init_context = get_init_weight_context_manager(
            use_meta_tensor=not model_config.tie_word_embeddings, mesh=self.device_mesh
        )

        # get dtype from config, default to bf16 for backward compatibility
        from verl.utils.torch_dtypes import PrecisionType
        model_dtype_str = config.model.get("dtype", "bf16")
        model_dtype = PrecisionType.to_dtype(model_dtype_str)

        with init_context(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model_config.classifier_dropout = 0.0
            model_config.hidden_dropout = "0"
            reward_module = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name_or_path=local_path,
                config=model_config,
                torch_dtype=model_dtype,
                attn_implementation="flash_attention_2",
                trust_remote_code=trust_remote_code,
            )

            apply_monkey_patch(
                model=reward_module,
                use_remove_padding=config.model.get("use_remove_padding", False),
                ulysses_sp_size=self.ulysses_sequence_parallel_size,
            )

            reward_module.to(model_dtype)

        auto_wrap_policy = get_fsdp_wrap_policy(module=reward_module, config=self.config.model.fsdp_config)

        fsdp_mesh = self.device_mesh
        sharding_strategy = get_sharding_strategy(fsdp_mesh)

        if config.strategy == "fsdp":
            reward_module = FSDP(
                reward_module,
                param_init_fn=init_fn,
                use_orig_params=False,
                auto_wrap_policy=auto_wrap_policy,
                device_id=get_device_id(),
                sharding_strategy=sharding_strategy,  # zero3
                sync_module_states=True,
                cpu_offload=CPUOffload(offload_params=True),
                forward_prefetch=self.config.model.fsdp_config.forward_prefetch,
                device_mesh=self.device_mesh,
            )
        elif config.strategy == "fsdp2":
            assert CPUOffloadPolicy is not None, "PyTorch version >= 2.4 is required for using fully_shard API (FSDP2)"
            cpu_offload = CPUOffloadPolicy(pin_memory=True)
            fsdp_kwargs = {
                "mesh": fsdp_mesh,
                "offload_policy": cpu_offload,
                "reshard_after_forward": config.model.fsdp_config.reshard_after_forward,
                "shard_placement_fn": get_shard_placement_fn(fsdp_size=self.device_mesh.shape[-1]),
            }
            full_state = reward_module.state_dict()
            apply_fsdp2(reward_module, fsdp_kwargs, config.model.fsdp_config)
            fsdp2_load_full_state_dict(reward_module, full_state, fsdp_mesh, cpu_offload)
        else:
            raise NotImplementedError(f"Unknown strategy: {config.strategy}")
        return reward_module

    def _compute_entropy_safe(self, logits, chunk_size=4096):
        import torch.nn.functional as F
        # logits: [..., vocab_size]
        original_shape = logits.shape
        vocab_size = original_shape[-1]
        
        # Flatten to [-1, vocab_size]
        logits_flat = logits.view(-1, vocab_size)
        
        entropy_list = []
        for i in range(0, logits_flat.size(0), chunk_size):
            chunk = logits_flat[i : i + chunk_size]
            # using log_softmax is more numerically stable
            log_probs = F.log_softmax(chunk, dim=-1)
            probs = torch.exp(log_probs)
            # Entropy = -sum(p * log(p))
            entropy = -torch.sum(probs * log_probs, dim=-1)
            entropy_list.append(entropy)
            
        entropy_flat = torch.cat(entropy_list, dim=0)
        
        # Reshape back to original shape minus vocab dim
        return entropy_flat.view(original_shape[:-1])

    def _compute_teacher_top_k_log_probs(self, logits, student_ids, top_k, strategy="only_stu", chunk_size=1024):
        # logits: (N, Vocab)
        # student_ids: (N, K_s)
        # output: (N, K_s)
        
        n_samples = logits.size(0)
        results = []
        valid_counts_list = []
        # Added overlap count list
        overlap_counts_list = []
        teacher_top_k_ids_list = []
        teacher_top_k_log_probs_list = []
        # For union strategy: teacher_in_student mask (T_in_S)
        teacher_in_student_list = []
        
        for start in range(0, n_samples, chunk_size):
            end = min(start + chunk_size, n_samples)
            
            logits_chunk = logits[start:end] # (chunk, Vocab)
            student_ids_chunk = student_ids[start:end] # (chunk, K_s)
            
            # 1. Top-K on Teacher
            # We always need teacher top-k for metrics (overlap)
            t_logits, t_ids = torch.topk(logits_chunk, k=top_k, dim=-1) # (chunk, K_t)
            t_logsumexp = torch.logsumexp(logits_chunk, dim=-1, keepdim=True)
            t_log_probs_top_k = t_logits - t_logsumexp

            teacher_top_k_ids_list.append(t_ids)
            teacher_top_k_log_probs_list.append(t_log_probs_top_k)
            
            # 2. Compute Intersection / Overlap (chunked to avoid OOM)
            # Expand for broadcasting
            s_ids_exp = student_ids_chunk.unsqueeze(-1) # (chunk, K_s, 1)
            t_ids_exp = t_ids.unsqueeze(-2) # (chunk, 1, K_t)
            
            matches = (s_ids_exp == t_ids_exp) # (chunk, K_s, K_t)
            
            # Count overlaps for metrics
            # matches is boolean, sum over K_t then K_s to get total overlap per sample?
            # Or we want to know which student tokens are in teacher top k.
            # matches.any(dim=-1) gives (chunk, K_s) boolean "is this student token in teacher top k?"
            is_in_teacher = matches.any(dim=-1) # (chunk, K_s)
            
            # Return mask (float for easier padding/handling)
            overlap_mask_chunk = is_in_teacher.float()
            overlap_counts_list.append(overlap_mask_chunk)
            
            # For union strategy: compute T_in_S (teacher id in student top k)
            # matches: (chunk, K_s, K_t), swap to get (chunk, K_t, K_s) then any over K_s
            is_in_student = matches.any(dim=-2) # (chunk, K_t) - teacher token in student top k
            teacher_in_student_list.append(is_in_student.float())

            if strategy in ["only_stu", "union","union-intersection"]:
                # For only_stu and union, we use all student tokens.
                # Just gather log probs for student_ids from full teacher distribution
                # We can do this efficiently without limited to top-k
                # But since we are in a chunk loop, we can do it here.
                
                # Re-select for this chunk
                # We need LogProbs not Logits
                
                chunk_log_probs = torch.gather(logits_chunk, dim=-1, index=student_ids_chunk) - t_logsumexp
                
                results.append(chunk_log_probs)
                
                # Valid counts is just K_s for everyone
                chunk_valid_counts = torch.full((logits_chunk.size(0),), student_ids_chunk.size(-1), device=logits.device, dtype=torch.long)
                valid_counts_list.append(chunk_valid_counts)

            else:
                # "intersection" strategy (and others if any)
                t_vals_exp = t_log_probs_top_k.unsqueeze(-2) # (chunk, 1, K_t)
                
                # Select values
                # If match, take value. If no match, use -inf.
                vals_masked = t_vals_exp.masked_fill(~matches, float('-inf'))
                
                # Reduce: Max over K_t dimension to get value for each Student Token
                # If a student token is not in Teacher Top K, all values are -inf, so max is -inf.
                chunk_res, _ = vals_masked.max(dim=-1) # (chunk, K_s)
                
                results.append(chunk_res)
                
                valid_counts_list.append(overlap_mask_chunk.sum(dim=-1).long())
            
        return (torch.cat(results, dim=0), torch.cat(valid_counts_list, dim=0), torch.cat(overlap_counts_list, dim=0),
                torch.cat(teacher_top_k_ids_list, dim=0), torch.cat(teacher_top_k_log_probs_list, dim=0),
                torch.cat(teacher_in_student_list, dim=0))

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        # This is used to import external_lib into the huggingface systems
        import_external_libs(self.config.model.get("external_lib", None))
        self.reward_module = self._build_model(config=self.config)

    def _forward_micro_batch(self, micro_batch, student_top_k_ids=None, compute_entropy=False, top_k=0, strategy="only_stu", teacher_temperature=1.0):
        from verl.utils.attention_utils import index_first_axis, pad_input, rearrange, unpad_input
        from verl.utils.ulysses import gather_outputs_and_unpad, ulysses_pad_and_slice_inputs, ulysses_pad
        import verl.utils.torch_functional as verl_F
        response_length = micro_batch["responses"].size(-1)
        with torch.no_grad(), torch.autocast(device_type=get_device_name(), dtype=torch.bfloat16):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 3, seqlen) -> (3, bsz, seqlen)
            
            teacher_on_student_log_probs = None
            teacher_top_k_ids = None
            teacher_top_k_log_probs = None
            teacher_entropy = None
            teacher_valid_counts = None
            teacher_overlap_mask = None
            teacher_in_student_mask = None  # For union strategy: T_in_S computed in chunks
            need_logits = student_top_k_ids is not None or compute_entropy

            if self.use_remove_padding:
                input_ids_rmpad, indices, *_ = unpad_input(
                    input_ids.unsqueeze(-1), attention_mask
                )  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = (
                        index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
                        .transpose(0, 1)
                        .unsqueeze(1)
                    )  # (3, bsz, seqlen) -> (3, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(
                        rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                    ).transpose(0, 1)

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    is_vlm_model = hasattr(
                        getattr(self.actor_module, "module", self.actor_module).config, "vision_config"
                    )
                    if is_vlm_model:
                        # vlm model's inputs will be sliced after embedding
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    else:
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad_rolled,
                        position_ids_rmpad=None,
                        sp_size=self.ulysses_sequence_parallel_size,
                    )

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)
                output = self.reward_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    use_cache=False,
                    return_dict=self.use_fused_kernels,
                )

                local_entropy_rmpad = None
                local_valid_counts = None
                local_overlap_mask = None
                local_top_k_log_probs_on_student_ids = None
                local_teacher_top_k_ids = None
                local_teacher_top_k_log_probs = None
                local_teacher_in_student_mask = None

                if self.use_fused_kernels and not need_logits:
                    rm_log_probs = output.log_probs.squeeze(0)  # (total_nnz,)
                    rm_log_probs = rm_log_probs.to(torch.float32)
                else:
                    logits_rmpad = output[0] if isinstance(output, tuple) else output.logits
                    logits_rmpad = logits_rmpad.squeeze(0)  # (total_nnz, vocab_size)
                    logits_rmpad = logits_rmpad.div_(teacher_temperature)

                    # Compute entropy if logits are available
                    # We compute entropy on the logits.
                    # Note: logits_rmpad is for the next token prediction.
                    # We want entropy at each position.
                    if compute_entropy:
                        local_entropy_rmpad = self._compute_entropy_safe(logits_rmpad) # (total_nnz,)

                    # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                    inplace_backward = True

                    rm_log_probs = verl_F.logprobs_from_logits(
                        logits=logits_rmpad,
                        labels=input_ids_rmpad_rolled,
                        inplace_backward=inplace_backward,
                    )
                    
                    if student_top_k_ids is not None:
                         k_s = student_top_k_ids.shape[-1]
                         # construct the full top_k_ids to align with the packed logits
                         # logits predict the next token, so we need to align the id at t+1 to logits at t
                         # response is at the end of the sequence
                         full_student_top_k_ids = torch.zeros((batch_size, seqlen, k_s), dtype=torch.long, device=logits_rmpad.device)
                         full_student_top_k_ids[:, -response_length-1:-1, :] = student_top_k_ids
                         
                         # pack it
                         packed_top_k_ids = index_first_axis(rearrange(full_student_top_k_ids, "b s k -> (b s) k"), indices)
                         
                         # slice it if sp
                         if self.use_ulysses_sp:
                             local_top_k_ids, _, _ = ulysses_pad_and_slice_inputs(
                                 packed_top_k_ids,
                                 position_ids_rmpad=None,
                                 sp_size=self.ulysses_sequence_parallel_size,
                             )
                         else:
                             local_top_k_ids = packed_top_k_ids

                         # compute local top_k log probs
                         # (total_nnz, k) or (total_nnz/sp, k)
                         
                         # Apply filtering if top_k > 0
                         if top_k > 0:
                             # Use chunked broadcasting to save memory
                             # logits_rmpad: (total_nnz, Vocab)
                             # local_top_k_ids: (total_nnz, K_s)
                            local_top_k_log_probs_on_student_ids, local_valid_counts, local_overlap_mask, local_teacher_top_k_ids, local_teacher_top_k_log_probs, local_teacher_in_student_mask = self._compute_teacher_top_k_log_probs(
                                logits=logits_rmpad,
                                student_ids=local_top_k_ids,
                                top_k=top_k,
                                strategy=strategy
                            )

                    else:
                         pass

                # pad it back
                if self.use_ulysses_sp:
                    full_log_probs = gather_outputs_and_unpad(
                        local_hidden_states=rm_log_probs.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                        sp_size=self.ulysses_sequence_parallel_size,
                        group=self.ulysses_sequence_parallel_group
                    )
                    if local_top_k_log_probs_on_student_ids is not None:
                        full_top_k_log_probs_on_student_ids = gather_outputs_and_unpad(
                            local_hidden_states=local_top_k_log_probs_on_student_ids,
                            indices=indices,
                            batch=batch_size,
                            seqlen=seqlen,
                            sp_size=self.ulysses_sequence_parallel_size,
                            group=self.ulysses_sequence_parallel_group
                        )
                    if local_teacher_top_k_ids is not None:
                         full_teacher_top_k_ids = gather_outputs_and_unpad(
                            local_hidden_states=local_teacher_top_k_ids,
                            indices=indices,
                            batch=batch_size,
                            seqlen=seqlen,
                            sp_size=self.ulysses_sequence_parallel_size,
                            group=self.ulysses_sequence_parallel_group
                         )
                    if local_teacher_top_k_log_probs is not None:
                         full_teacher_top_k_log_probs = gather_outputs_and_unpad(
                            local_hidden_states=local_teacher_top_k_log_probs,
                            indices=indices,
                            batch=batch_size,
                            seqlen=seqlen,
                            sp_size=self.ulysses_sequence_parallel_size,
                            group=self.ulysses_sequence_parallel_group
                         )
                    if local_valid_counts is not None:
                        full_valid_counts = gather_outputs_and_unpad(
                            local_hidden_states=local_valid_counts.unsqueeze(-1),
                            indices=indices,
                            batch=batch_size,
                            seqlen=seqlen,
                            sp_size=self.ulysses_sequence_parallel_size,
                            group=self.ulysses_sequence_parallel_group
                        )
                    if local_overlap_mask is not None:
                        full_overlap_mask = gather_outputs_and_unpad(
                            local_hidden_states=local_overlap_mask.unsqueeze(-1), # Mask is already 1D per token? No, mask is (NNZ, K), wait.
                            # compute_teacher_top_k_log_probs return cat(results)
                            # results is (chunk, K_s). So local_overlap_mask is (NNZ, K_s).
                            indices=indices,
                            batch=batch_size,
                            seqlen=seqlen,
                            sp_size=self.ulysses_sequence_parallel_size,
                            group=self.ulysses_sequence_parallel_group
                        )
                    if local_teacher_in_student_mask is not None:
                        full_teacher_in_student_mask = gather_outputs_and_unpad(
                            local_hidden_states=local_teacher_in_student_mask,
                            indices=indices,
                            batch=batch_size,
                            seqlen=seqlen,
                            sp_size=self.ulysses_sequence_parallel_size,
                            group=self.ulysses_sequence_parallel_group
                        )
                    if local_entropy_rmpad is not None:
                        full_entropy = gather_outputs_and_unpad(
                            local_hidden_states=local_entropy_rmpad.unsqueeze(-1),
                            indices=indices,
                            batch=batch_size,
                            seqlen=seqlen,
                            sp_size=self.ulysses_sequence_parallel_size,
                            group=self.ulysses_sequence_parallel_group
                        )
                else:
                    full_log_probs = pad_input(
                        hidden_states=rm_log_probs.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                    if local_top_k_log_probs_on_student_ids is not None:
                        full_top_k_log_probs_on_student_ids = pad_input(
                            hidden_states=local_top_k_log_probs_on_student_ids,
                            indices=indices,
                            batch=batch_size,
                            seqlen=seqlen,
                        )
                    if local_teacher_top_k_ids is not None:
                        full_teacher_top_k_ids = pad_input(
                            hidden_states=local_teacher_top_k_ids,
                            indices=indices,
                            batch=batch_size,
                            seqlen=seqlen,
                        )
                    if local_teacher_top_k_log_probs is not None:
                        full_teacher_top_k_log_probs = pad_input(
                            hidden_states=local_teacher_top_k_log_probs,
                            indices=indices,
                            batch=batch_size,
                            seqlen=seqlen,
                        )
                    if local_valid_counts is not None:
                        full_valid_counts = pad_input(
                            hidden_states=local_valid_counts.unsqueeze(-1),
                            indices=indices,
                            batch=batch_size,
                            seqlen=seqlen,
                        )
                    if local_overlap_mask is not None:
                        # local_overlap_mask is (NNZ, K_s)
                        # pad_input expects (NNZ, Hidden). K_s is "Hidden".
                        # So we don't unsqueeze if K_s > 1. 
                        # But wait, logic suggests unsqueeze(-1) for scalar? 
                        # Previous code: local_overlap_counts was (NNZ,), so unsqueeze(-1) -> (NNZ, 1).
                        # varying K? "strategy=only_stu" -> K_s.
                        # If K_s > 1, no unsqueeze.
                        # pad_input pads dim 0.
                        full_overlap_mask = pad_input(
                            hidden_states=local_overlap_mask, 
                            indices=indices,
                            batch=batch_size,
                            seqlen=seqlen,
                        )
                    if local_teacher_in_student_mask is not None:
                        full_teacher_in_student_mask = pad_input(
                            hidden_states=local_teacher_in_student_mask,
                            indices=indices,
                            batch=batch_size,
                            seqlen=seqlen,
                        )
                    if local_entropy_rmpad is not None:
                        full_entropy = pad_input(
                            hidden_states=local_entropy_rmpad.unsqueeze(-1),
                            indices=indices,
                            batch=batch_size,
                            seqlen=seqlen,
                        )

                rm_log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
                if local_top_k_log_probs_on_student_ids is not None:
                    teacher_on_student_log_probs = full_top_k_log_probs_on_student_ids[:, -response_length - 1 : -1, :]
                if local_teacher_top_k_ids is not None:
                    teacher_top_k_ids = full_teacher_top_k_ids[:, -response_length - 1 : -1, :]
                if local_teacher_top_k_log_probs is not None:
                    teacher_top_k_log_probs = full_teacher_top_k_log_probs[:, -response_length - 1 : -1, :]
                if local_valid_counts is not None:
                    teacher_valid_counts = full_valid_counts.squeeze(-1)[:, -response_length - 1 : -1]
                if local_overlap_mask is not None:
                    teacher_overlap_mask = full_overlap_mask[:, -response_length - 1 : -1, :] # Keep K dimension
                if local_teacher_in_student_mask is not None:
                    teacher_in_student_mask = full_teacher_in_student_mask[:, -response_length - 1 : -1, :] # Keep K dimension
                if local_entropy_rmpad is not None:
                    teacher_entropy = full_entropy.squeeze(-1)[:, -response_length - 1 : -1]

            else:
                output = self.reward_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    use_cache=False,
                    return_dict=self.use_fused_kernels,
                )

                if self.use_fused_kernels and not need_logits:
                    rm_log_probs = output.log_probs[:, :-1]  # (bsz, seq_length)
                    rm_log_probs = rm_log_probs.to(torch.float32)

                    # pad rm_log_probs to full sequence length
                    full_rm_log_probs = torch.zeros_like(input_ids, dtype=rm_log_probs.dtype)
                    full_rm_log_probs[:, :-1] = rm_log_probs
                    rm_log_probs = full_rm_log_probs
                else:
                    # When return_dict=False, output is a tuple with logits as first element
                    rm_output_logits = output[0] if isinstance(output, tuple) else output.logits
                    rm_logits_resp = rm_output_logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
                    rm_logits_resp = rm_logits_resp.div_(teacher_temperature)
                    
                    # Compute entropy
                    if compute_entropy:
                        teacher_entropy = self._compute_entropy_safe(rm_logits_resp) # (bsz, response_length)

                    rm_log_probs = verl_F.logprobs_from_logits(rm_logits_resp, micro_batch["responses"])
                    
                    if student_top_k_ids is not None:
                        if top_k > 0:
                            # Use chunked broadcasting to save memory
                            # flatten inputs
                            # rm_logits_resp: (bsz, seqlen, Vocab)
                            # student_top_k_ids: (bsz, seqlen, K_s)
                            
                            original_shape = student_top_k_ids.shape # (bsz, seqlen, K_s)
                            
                            flat_logits = rm_logits_resp.reshape(-1, rm_logits_resp.size(-1)) # (N, Vocab)
                            flat_ids = student_top_k_ids.reshape(-1, student_top_k_ids.size(-1)) # (N, K_s)
                            
                            # Pass strategy from meta_info
                            flat_on_student_log_probs, flat_counts, flat_overlap_mask, flat_teacher_top_k_ids, flat_teacher_top_k_log_probs, flat_teacher_in_student = self._compute_teacher_top_k_log_probs(
                                logits=flat_logits,
                                student_ids=flat_ids,
                                top_k=top_k,
                                strategy=strategy
                            )
                            
                            teacher_on_student_log_probs = flat_on_student_log_probs.view(original_shape)
                            teacher_valid_counts = flat_counts.view(original_shape[:-1])
                            teacher_overlap_mask = flat_overlap_mask.view(original_shape) # (Batch, Seq, K)
                            teacher_top_k_ids = flat_teacher_top_k_ids.view(original_shape[0], original_shape[1], top_k)
                            teacher_top_k_log_probs = flat_teacher_top_k_log_probs.view(original_shape[0], original_shape[1], top_k)
                            teacher_in_student_mask = flat_teacher_in_student.view(original_shape[0], original_shape[1], top_k)
                            
                        else:
                            teacher_on_student_log_probs = torch.gather(rm_logits_resp, dim=-1, index=student_top_k_ids)
                            teacher_logsumexp = torch.logsumexp(rm_logits_resp, dim=-1, keepdim=True)
                            teacher_on_student_log_probs = teacher_on_student_log_probs - teacher_logsumexp
                            teacher_overlap_mask = None

            return rm_log_probs, teacher_on_student_log_probs, teacher_top_k_ids, teacher_top_k_log_probs, teacher_entropy, teacher_valid_counts, teacher_overlap_mask, teacher_in_student_mask

    def _expand_to_token_level(self, data: DataProto, scores: torch.Tensor):
        batch_size = data.batch.batch_size[0]
        # expand as token_level_reward
        attention_mask = data.batch["attention_mask"]
        position_ids = data.batch["position_ids"]
        response_length = data.batch["responses"].shape[-1]
        if position_ids.dim() == 3:  # qwen2vl mrope [bs, 3, seq_len]
            position_ids = position_ids[:, 0, :]
        eos_mask_idx = torch.argmax(position_ids * attention_mask, dim=-1)  # (bsz,)
        token_level_scores = torch.zeros_like(attention_mask, dtype=scores.dtype)  # (bsz, seqlen)
        token_level_scores[torch.arange(batch_size), eos_mask_idx] = scores

        # select the response part
        token_level_scores = token_level_scores[:, -response_length:]

        return token_level_scores

    def _switch_chat_template(self, data: DataProto):
        src_max_length = data.batch["attention_mask"].shape[-1]

        src_tokenizer = self.input_tokenizer
        target_tokenizer = self.tokenizer

        rm_input_ids = []
        rm_attention_mask = []

        for i in range(data.batch.batch_size[0]):
            if not isinstance(data.non_tensor_batch["raw_prompt"][i], list | np.ndarray):
                raise TypeError(
                    f"raw_prompt must be a list or numpy array, got {type(data.non_tensor_batch['raw_prompt'][i])}"
                )

            # extract raw prompt
            chat: list = list(data.non_tensor_batch["raw_prompt"][i])

            # extract response
            response_ids = data.batch["responses"][i]
            response_length = response_ids.shape[-1]
            valid_response_length = data.batch["attention_mask"][i][-response_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            response = src_tokenizer.decode(valid_response_ids)
            # remove bos and eos
            response = response.replace(src_tokenizer.eos_token, "")

            chat.append({"role": "assistant", "content": response})

            prompt_with_chat_template = target_tokenizer.apply_chat_template(
                chat, add_generation_prompt=False, tokenize=False
            )
            # the maximum length is actually determined by the reward model itself
            max_length = self.config.get("max_length", src_max_length)
            if max_length is None:
                max_length = src_max_length

            model_inputs = target_tokenizer(prompt_with_chat_template, return_tensors="pt", add_special_tokens=False)
            input_ids, attention_mask = verl_F.postprocess_data(
                input_ids=model_inputs["input_ids"],
                attention_mask=model_inputs["attention_mask"],
                max_length=max_length,
                pad_token_id=target_tokenizer.pad_token_id,
                left_pad=False,  # right padding
                truncation=self.config.get("truncation", "right"),
            )  # truncate from the right

            rm_input_ids.append(input_ids)
            rm_attention_mask.append(attention_mask)

        rm_input_ids = torch.cat(rm_input_ids, dim=0)
        rm_attention_mask = torch.cat(rm_attention_mask, dim=0)

        rm_position_ids = compute_position_id_with_mask(rm_attention_mask)

        rm_inputs = {"input_ids": rm_input_ids, "attention_mask": rm_attention_mask, "position_ids": rm_position_ids}

        return DataProto.from_dict(rm_inputs)

    def _switch_chat_template_token_level(self, data: DataProto):
        """Re-tokenize with the RM's chat template while preserving token-level alignment.

        Unlike _switch_chat_template (which uses right-padding for sequence-level RM),
        this function uses LEFT-padding so that the response stays at the end of the
        sequence. This ensures that `[-response_length-1:-1]` correctly targets the
        response positions for token-level distillation (student_top_k_ids alignment).

        Requirements: actor and reward model must share the same vocabulary.
        """
        src_max_length = data.batch["attention_mask"].shape[-1]

        src_tokenizer = self.input_tokenizer
        target_tokenizer = self.tokenizer

        is_debug = (self.rank == 0)  # only print on rank 0

        if is_debug:
            print(f"\n{'='*80}")
            print(f"[DEBUG _switch_chat_template_token_level] START")
            print(f"  src_tokenizer: {type(src_tokenizer).__name__}, vocab_size={src_tokenizer.vocab_size}")
            print(f"  target_tokenizer: {type(target_tokenizer).__name__}, vocab_size={target_tokenizer.vocab_size}")
            print(f"  src_max_length={src_max_length}")
            print(f"  batch_size={data.batch.batch_size[0]}")
            print(f"  original input_ids shape: {data.batch['input_ids'].shape}")
            print(f"  original attention_mask shape: {data.batch['attention_mask'].shape}")
            print(f"  original responses shape: {data.batch['responses'].shape}")
            print(f"{'='*80}")

        rm_input_ids = []
        rm_attention_mask = []
        rm_responses = []

        for i in range(data.batch.batch_size[0]):
            if not isinstance(data.non_tensor_batch["raw_prompt"][i], list | np.ndarray):
                raise TypeError(
                    f"raw_prompt must be a list or numpy array, got {type(data.non_tensor_batch['raw_prompt'][i])}"
                )

            # extract raw prompt
            chat: list = list(data.non_tensor_batch["raw_prompt"][i])

            # extract response
            response_ids = data.batch["responses"][i]
            response_length = response_ids.shape[-1]
            valid_response_length = data.batch["attention_mask"][i][-response_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode using the actor's tokenizer
            response = src_tokenizer.decode(valid_response_ids)
            # remove bos and eos
            if src_tokenizer.eos_token:
                response = response.replace(src_tokenizer.eos_token, "")

            if is_debug and i == 0:
                print(f"\n[DEBUG sample 0] --- Response decode/re-encode ---")
                print(f"  original response_ids shape: {response_ids.shape}")
                print(f"  response_length (padded): {response_length}")
                print(f"  valid_response_length: {valid_response_length}")
                print(f"  original response_ids (first 20): {valid_response_ids[:20].tolist()}")
                print(f"  decoded response (first 200 chars): {response[:200]}")

            chat.append({"role": "assistant", "content": response})

            prompt_with_chat_template = target_tokenizer.apply_chat_template(
                chat, add_generation_prompt=False, tokenize=False
            )

            if is_debug and i == 0:
                print(f"  chat template applied (first 300 chars): {prompt_with_chat_template[:300]}")
                print(f"  chat template applied (last 200 chars): {prompt_with_chat_template[-200:]}")

            max_length = self.config.get("max_length", src_max_length)
            if max_length is None:
                max_length = src_max_length

            model_inputs = target_tokenizer(prompt_with_chat_template, return_tensors="pt", add_special_tokens=False)

            if is_debug and i == 0:
                raw_len = model_inputs["input_ids"].shape[-1]
                print(f"  re-tokenized length (before pad/trunc): {raw_len}")
                print(f"  max_length for postprocess: {max_length}")

            input_ids, attention_mask = verl_F.postprocess_data(
                input_ids=model_inputs["input_ids"],
                attention_mask=model_inputs["attention_mask"],
                max_length=max_length,
                pad_token_id=target_tokenizer.pad_token_id,
                left_pad=True,  # LEFT padding to keep response at end
                truncation=self.config.get("truncation", "left"),  # truncate prompt from left if needed
            )

            if is_debug and i == 0:
                content_len = attention_mask.sum().item()
                pad_len = max_length - content_len
                # find where content starts (first non-pad position)
                first_content_pos = (attention_mask.squeeze(0) == 1).nonzero(as_tuple=True)[0]
                first_pos = first_content_pos[0].item() if len(first_content_pos) > 0 else -1
                last_pos = first_content_pos[-1].item() if len(first_content_pos) > 0 else -1
                print(f"  after postprocess: input_ids shape={input_ids.shape}")
                print(f"  content_len={content_len}, pad_len={pad_len}")
                print(f"  content range: [{first_pos}, {last_pos}]")
                print(f"  last 10 input_ids: {input_ids.squeeze(0)[-10:].tolist()}")
                print(f"  last 10 attn_mask: {attention_mask.squeeze(0)[-10:].tolist()}")

            rm_input_ids.append(input_ids)
            rm_attention_mask.append(attention_mask)

            # Re-tokenize the response alone with the target tokenizer to get correct response_ids
            response_inputs = target_tokenizer(response, return_tensors="pt", add_special_tokens=False)
            new_response_ids = response_inputs["input_ids"].squeeze(0)  # (new_resp_len,)

            if is_debug and i == 0:
                print(f"\n[DEBUG sample 0] --- Response re-tokenization ---")
                print(f"  new_response_ids length: {new_response_ids.shape[0]}")
                print(f"  original valid_response_length: {valid_response_length}")
                print(f"  target response_length (padded): {response_length}")
                print(f"  new_response_ids (first 20): {new_response_ids[:20].tolist()}")
                print(f"  orig valid_response_ids (first 20): {valid_response_ids[:20].tolist()}")
                # Check token-by-token match
                min_len = min(new_response_ids.shape[0], valid_response_ids.shape[0])
                match_count = (new_response_ids[:min_len] == valid_response_ids[:min_len].cpu()).sum().item()
                print(f"  token match in first {min_len} tokens: {match_count}/{min_len}")
                if match_count < min_len:
                    # Find first mismatch
                    for j in range(min_len):
                        if new_response_ids[j] != valid_response_ids[j].cpu():
                            print(f"  FIRST MISMATCH at pos {j}: new={new_response_ids[j].item()} "
                                  f"('{target_tokenizer.decode([new_response_ids[j].item()])}') vs "
                                  f"orig={valid_response_ids[j].item()} "
                                  f"('{src_tokenizer.decode([valid_response_ids[j].item()])}')")
                            break

            # Pad/truncate to match original response_length for alignment
            if new_response_ids.shape[0] >= response_length:
                if is_debug and i == 0:
                    print(f"  -> TRUNCATING new_response_ids from {new_response_ids.shape[0]} to {response_length}")
                # truncate to original response_length
                new_response_ids = new_response_ids[:response_length]
            else:
                pad_size = response_length - new_response_ids.shape[0]
                if is_debug and i == 0:
                    print(f"  -> PADDING new_response_ids from {new_response_ids.shape[0]} by {pad_size} to {response_length}")
                # right-pad with pad_token_id
                new_response_ids = torch.cat([
                    new_response_ids,
                    torch.full((pad_size,), target_tokenizer.pad_token_id, dtype=new_response_ids.dtype)
                ])
            rm_responses.append(new_response_ids.unsqueeze(0))

        rm_input_ids = torch.cat(rm_input_ids, dim=0)
        rm_attention_mask = torch.cat(rm_attention_mask, dim=0)
        rm_responses = torch.cat(rm_responses, dim=0)

        rm_position_ids = compute_position_id_with_mask(rm_attention_mask)

        if is_debug:
            print(f"\n[DEBUG _switch_chat_template_token_level] FINAL SHAPES:")
            print(f"  rm_input_ids: {rm_input_ids.shape}")
            print(f"  rm_attention_mask: {rm_attention_mask.shape}")
            print(f"  rm_position_ids: {rm_position_ids.shape}")
            print(f"  rm_responses: {rm_responses.shape}")
            # Verify response is at the end for first sample
            resp_len = rm_responses.shape[-1]
            last_tokens = rm_input_ids[0, -resp_len:].tolist()
            resp_tokens = rm_responses[0].tolist()
            # Check how many of the last resp_len tokens in input_ids match responses
            match = sum(1 for a, b in zip(last_tokens, resp_tokens) if a == b)
            print(f"  response_length={resp_len}")
            print(f"  last {resp_len} input_ids tokens vs rm_responses match: {match}/{resp_len}")
            print(f"  last 10 of input_ids[0]: {rm_input_ids[0, -10:].tolist()}")
            print(f"  last 10 of rm_responses[0]: {rm_responses[0, -10:].tolist()}")
            print(f"{'='*80}\n")

        rm_inputs = {
            "input_ids": rm_input_ids,
            "attention_mask": rm_attention_mask,
            "position_ids": rm_position_ids,
            "responses": rm_responses,
        }

        return DataProto.from_dict(rm_inputs)

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="reward"))
    @DistProfiler.annotate(color="brown")
    def compute_rm_score(self, data: DataProto, kl_estimator="k1"):
        import itertools

        from verl.utils.seqlen_balancing import get_reverse_idx, rearrange_micro_batches

        # Support all hardwares
        data = data.to(get_device_id())

        # Get student log probabilities from trajectories
        student_logp = data.batch["old_log_probs"]  # shape: [batch, response_len]
        
        student_top_k_ids = None
        student_top_k_log_probs = None
        if "student_top_k_ids" in data.batch.keys():
             student_top_k_ids = data.batch["student_top_k_ids"]
             student_top_k_log_probs = data.batch["student_top_k_log_probs"]
        
        # Get global_steps from meta_info
        global_steps = data.meta_info.get("global_steps", -1)
        is_plot = data.meta_info.get("is_plot", False)
        # Compute teacher entropy every step for logging, but only plot every 10 steps
        compute_entropy = True

        # Get response mask to identify valid (non-padded) response tokens

        response_mask = data.batch["response_mask"]  # shape: [batch, response_len]

        if self._do_switch_chat_template:
            if self.rank == 0:
                print(f"Chat template switching is ENABLED (token-level aligned, left-padded).")
            rm_data = self._switch_chat_template_token_level(data)
        else:
            rm_inputs = {
                "input_ids": data.batch["input_ids"],
                "attention_mask": data.batch["attention_mask"],
                "position_ids": data.batch["position_ids"],
                "responses": data.batch["responses"],
            }
            rm_data = DataProto.from_dict(rm_inputs)

        # Support all hardwares
        rm_data = rm_data.to(get_device_id())
        
        if student_top_k_ids is not None:
             rm_data.batch["student_top_k_ids"] = student_top_k_ids

        # perform forward computation
        with self.ulysses_sharding_manager:
            use_dynamic_bsz = self.config.use_dynamic_bsz
            if use_dynamic_bsz:
                max_token_len = self.config.forward_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                micro_batches, indices = rearrange_micro_batches(batch=rm_data.batch, max_token_len=max_token_len)
            else:
                micro_batches = rm_data.batch.split(self.config.micro_batch_size_per_gpu)
            
            # Get Top-K and Top-P from config
            top_k = data.meta_info.get("log_prob_top_k", self.config.get("log_prob_top_k", 0))
            top_k_strategy = data.meta_info.get("top_k_strategy", self.config.get("top_k_strategy", "only_stu"))
            teacher_temperature = data.meta_info.get("teacher_temperature", self.config.get("teacher_temperature", 1.0))
            
            output_logp = []
            output_on_student_logp = []
            output_teacher_top_k_ids = []
            output_teacher_top_k_logp = []
            output_entropy = []
            output_valid_counts = []
            output_overlap_counts = []
            output_teacher_in_student = []  # For union strategy: T_in_S computed in chunks
            
            for micro_batch in micro_batches:
                # micro_batch is a DataProto or DataProtoItem.
                # If it's a DataProto, it has .batch (TensorDict).
                # If it's a TensorDict (from .split()), it behaves like a dict.
                
                # Check if micro_batch is a DataProto or DataProtoItem
                if hasattr(micro_batch, 'batch') and isinstance(micro_batch.batch, TensorDict):
                    mb_top_k_ids = micro_batch.batch.get("student_top_k_ids", None)
                elif isinstance(micro_batch, TensorDict):
                    # Direct TensorDict
                    mb_top_k_ids = micro_batch.get("student_top_k_ids", None)
                else:
                    # Fallback for other types (e.g. dict) if split behaves differently
                    mb_top_k_ids = micro_batch.get("student_top_k_ids", None) if hasattr(micro_batch, "get") else None

                teacher_logp_batch, teacher_on_student_logp_batch, teacher_top_k_ids_batch, teacher_top_k_logp_teacher_batch, teacher_entropy_batch, teacher_valid_counts_batch, teacher_overlap_mask_batch, teacher_in_student_mask_batch = self._forward_micro_batch(
                    micro_batch, 
                    student_top_k_ids=mb_top_k_ids,
                    compute_entropy=compute_entropy,
                    top_k=top_k,
                    strategy=top_k_strategy,
                    teacher_temperature=teacher_temperature
                )
                output_logp.append(teacher_logp_batch)
                if teacher_on_student_logp_batch is not None:
                    output_on_student_logp.append(teacher_on_student_logp_batch)
                if teacher_top_k_ids_batch is not None:
                    output_teacher_top_k_ids.append(teacher_top_k_ids_batch)
                if teacher_top_k_logp_teacher_batch is not None:
                    output_teacher_top_k_logp.append(teacher_top_k_logp_teacher_batch)
                if teacher_entropy_batch is not None:
                    output_entropy.append(teacher_entropy_batch)
                if teacher_valid_counts_batch is not None:
                    output_valid_counts.append(teacher_valid_counts_batch)
                if teacher_overlap_mask_batch is not None:
                    output_overlap_counts.append(teacher_overlap_mask_batch)
                if teacher_in_student_mask_batch is not None:
                    output_teacher_in_student.append(teacher_in_student_mask_batch)
                    
            teacher_logp = torch.cat(output_logp, dim=0)
            teacher_on_student_logp = None
            if len(output_on_student_logp) > 0:
                teacher_on_student_logp = torch.cat(output_on_student_logp, dim=0)
            
            teacher_top_k_ids = None
            if len(output_teacher_top_k_ids) > 0:
                teacher_top_k_ids = torch.cat(output_teacher_top_k_ids, dim=0)

            teacher_top_k_logp = None
            if len(output_teacher_top_k_logp) > 0:
                teacher_top_k_logp = torch.cat(output_teacher_top_k_logp, dim=0)
            
            teacher_entropy = None
            if len(output_entropy) > 0:
                teacher_entropy = torch.cat(output_entropy, dim=0)

            teacher_valid_counts = None
            if len(output_valid_counts) > 0:
                teacher_valid_counts = torch.cat(output_valid_counts, dim=0)

            teacher_overlap_mask = None
            if len(output_overlap_counts) > 0:
                teacher_overlap_mask = torch.cat(output_overlap_counts, dim=0)

            teacher_in_student_mask = None
            if len(output_teacher_in_student) > 0:
                teacher_in_student_mask = torch.cat(output_teacher_in_student, dim=0)

            if use_dynamic_bsz:
                indices = list(itertools.chain.from_iterable(indices))
                assert len(indices) == teacher_logp.size(0), f"{len(indices)} vs. {teacher_logp.size(0)}"
                revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long, device=teacher_logp.device)
                teacher_logp = teacher_logp[revert_indices]
                if teacher_on_student_logp is not None:
                    teacher_on_student_logp = teacher_on_student_logp[revert_indices]
                if teacher_top_k_ids is not None:
                    teacher_top_k_ids = teacher_top_k_ids[revert_indices]
                if teacher_top_k_logp is not None:
                    teacher_top_k_logp = teacher_top_k_logp[revert_indices]
                if teacher_entropy is not None:
                    teacher_entropy = teacher_entropy[revert_indices]
                if teacher_valid_counts is not None:
                    teacher_valid_counts = teacher_valid_counts[revert_indices]
                if teacher_overlap_mask is not None:
                    teacher_overlap_mask = teacher_overlap_mask[revert_indices]
                if teacher_in_student_mask is not None:
                    teacher_in_student_mask = teacher_in_student_mask[revert_indices]

            if top_k > 0:
                # Reward calculation is moved to ray_trainer for top_k > 0
                # because it needs student_on_teacher_log_probs which requires another actor forward
                rm_scores = None 
                overlap_mask = teacher_overlap_mask
            else:
                print("Top-k log probs not present, just using student_logp - teacher_logp as reward")
                
                reverse_kl = student_logp - teacher_logp
                rm_scores = -reverse_kl
                
                teacher_valid_counts = None
                overlap_mask = None
            
            tensors = {}
            if rm_scores is not None:
                tensors["rm_scores"] = rm_scores
            
            if teacher_on_student_logp is not None:
                tensors["teacher_on_student_log_probs"] = teacher_on_student_logp

            if teacher_top_k_ids is not None:
                tensors["teacher_top_k_ids"] = teacher_top_k_ids

            if teacher_top_k_logp is not None:
                tensors["teacher_top_k_log_probs"] = teacher_top_k_logp

            if teacher_entropy is not None:
                tensors["teacher_entropy"] = teacher_entropy
                
            if teacher_valid_counts is not None:
                tensors["teacher_valid_counts"] = teacher_valid_counts
            if overlap_mask is not None:
                tensors["overlap_mask"] = overlap_mask
            if teacher_in_student_mask is not None:
                tensors["teacher_in_student_mask"] = teacher_in_student_mask

            output = DataProto.from_dict(tensors=tensors)

        # https://pytorch.org/docs/stable/notes/fsdp.html#fsdp-notes
        # unshard the root FSDP module
        if self.world_size > 1 and fsdp_version(self.reward_module) == 1:
            self.reward_module._handle.reshard(True)

        # Keep on GPU to avoid expensive CPU-GPU transfer for large top-k data
        # output = output.to("cpu")
        return output


# ================================= Async related workers =================================
class AsyncActorRolloutRefWorker(ActorRolloutRefWorker):
    @register(dispatch_mode=Dispatch.DIRECT_ROLLOUT_METHOD)
    async def wake_up(self):
        await self.rollout_mode()
        return True

    @register(dispatch_mode=Dispatch.DIRECT_ROLLOUT_METHOD)
    async def sleep(self):
        await self.trainer_mode()
        return True

    # ============================ vLLM related ============================

    @register(dispatch_mode=Dispatch.DIRECT_ROLLOUT_METHOD)
    def get_zeromq_address(self):
        return self.rollout.get_zeromq_address()

    # ============================ SGLang related ============================

    @register(dispatch_mode=Dispatch.DIRECT_ROLLOUT_METHOD, blocking=False)
    async def chat_completion(self, json_request):
        ret = await self.rollout.chat_completion(json_request)
        return ret

    @register(dispatch_mode=Dispatch.DIRECT_ROLLOUT_METHOD, blocking=False)
    async def generate(
        self,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        request_id: str,
        image_data: Optional[list[Any]] = None,
    ) -> list[int]:
        ret = await self.rollout.generate(prompt_ids, sampling_params, request_id, image_data=image_data)
        return ret
