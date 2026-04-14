#!/bin/bash
#SBATCH --job-name=url
#SBATCH --output=logs/20251004/output_%j.log
#SBATCH --error=logs/20251004/error_%j.log
#SBATCH --account=test
#SBATCH --partition=TEST1
#SBATCH --exclude=g[81-82]
#SBATCH --gres=gpu:8
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=500G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

set -x

ray stop --force

export PYTHONUNBUFFERED=1
export PROJECT_NAME='OnPolicyDistillation' # TODO

# DeepMath-103K
export MAX_RESP_LENGTH= 8192 # TODO: 15360 / 7168 / 3072
export MAX_VAL_RESP_LENGTH= 16384 # TODO: 15360 / 7168 / 3072
export MINI_BATCH_SIZE=${MINI_BATCH_SIZE:-64} # TODO: 1 / 8 / 16 / 32 / 64 (default 64)
export TEMPERATURE=${TEMPERATURE:-1.0} # TODO: 0.6 / 0.8 / 1.0 / 1.2 (default 1.0)
export N_RESPONSES=32 # TODO: 4 / 8 / 16 / 32 (default: 8)
# export LR=${LR:-1e-6}s
# export LR_SCHEDULER=${LR_SCHEDULER:-constant}
export USE_KL=${USE_KL:-False} # TODO: True / False (default False)

# TODO: qwen3_1p7b_base / qwen3_1p7b / llama31_8b_base / llama31_8b_inst / qwen3_8b_base / qwen3_8b / qwen25_1p5b_base / qwen25_1p5b_inst / qwen25_7b_base / qwen25_7b_inst / qwen25_math_7b_base / qwen25_math_7b_inst / qwen25_math_1p5b_base / qwen25_math_1p5b_inst / distill_r1_1p5b / olmo2_1124_7b_base / olmo2_1124_7b_sft / olmo2_1124_7b_inst / llama32_3b_inst
export EXPERIMENT_NAME=token_reward_direct_DeepMath-103K_Qwen3-1.7B_8192-T_${TEMPERATURE}-n_${N_RESPONSES}-mbs_${MINI_BATCH_SIZE}-$(date +%Y-%m-%d_%H-%M-%S)
# export EXPERIMENT_NAME=grpo_${TASK}_llama31_tulu3_8b_sft_8k-T_${TEMPERATURE}-n_${N_RESPONSES}-kl_${USE_KL}-mbs_${MINI_BATCH_SIZE}-${REWARD_TYPE}-$(date +%Y-%m-%d_%H-%M-%S)

export TRAIN_DATASET= /home/test/test14/lyx_new/datasets/DeepMath-103K/verl_format/train.parquet
export TEST_DATA_DIR = /home/test/testdata/hbx/data
# TRAIN_DATASET=${TRAIN_FILE:-["$DATA_DIR/$TASK/train_${SAMPLE_SIZE}.parquet"]}
TEST_DATASET=${TEST_FILE:-["$DATA_DIR/AIME24/test.parquet","$DATA_DIR/AIME25/test.parquet","$DATA_DIR/AMC23/test.parquet"]}
# TEST_DATASET=${TEST_FILE:-["$DATA_DIR/AIME24/test.parquet","$DATA_DIR/AIME25/test.parquet","$DATA_DIR/AMC23/test.parquet","$DATA_DIR/MATH-500/test.parquet","$DATA_DIR/Minerva/test.parquet","$DATA_DIR/Olympiad-Bench/test.parquet"]}

# TODO:
# export ACTOR_MODEL_PATH=/home/test/test06/hbx/models/Qwen2.5-Math-1.5B
# export ACTOR_MODEL_PATH=/home/test/testdata/models/DeepSeek-R1-Distill-Qwen-1.5B
# export ACTOR_MODEL_PATH=/home/test/testdata/models/Qwen2.5-1.5B
# export ACTOR_MODEL_PATH=/home/test/testdata/models/Meta-Llama-3.1-8B-Instruct
# export ACTOR_MODEL_PATH=/home/test/testdata/models/Llama-3.2-3B-Instruct
# export ACTOR_MODEL_PATH=/home/test/test06/hbx/models/Qwen2.5-Math-7B
export ACTOR_MODEL_PATH=/home/test/test06/hbx/models/Qwen3-1.7B
# export ACTOR_MODEL_PATH=/home/test/testdata/models/Meta-Llama-3.1-8B
# export ACTOR_MODEL_PATH=/home/test/testdata/models/Qwen3-8B-Base

export PROJECT_PATH=/home/test/test14/lyx_new/OnPolicyDistillation
export PARALLEL_SIZE=1
export CKPT_PATH=${PROJECT_PATH}/checkpoint
export OUTLINES_CACHE_DIR=~/.cache/outlines/$(uuidgen)
export NCCL_DEBUG=WARN

# export VLLM_ATTENTION_BACKEND=XFORMERS
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=true
export SWANLAB_LOG_DIR=${PROJECT_PATH}/swanlab/${PROJECT_NAME}/
export HYDRA_FULL_ERROR=1

unset ROCR_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# TODO: DEBUG
# 1. temperature, higher
# 2. mini bs, on-policy
# 3. sharper distribution may work better on simple dataset?

KL_ARGS=""
if [ "$USE_KL" = "True" ]; then
    KL_ARGS="actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.005 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl"
else
    KL_ARGS="actor_rollout_ref.actor.use_kl_loss=False"
fi

LR_ARGS=""
if [ "$LR_SCHEDULER" = "cosine" ]; then
    LR_ARGS="actor_rollout_ref.actor.optim.warmup_style=cosine \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.03"
fi

PPO_MAX_TOKEN_LEN_PER_GPU=$(( ((1024 + MAX_RESP_LENGTH) > 32768) ? (1024 + MAX_RESP_LENGTH) : 32768))
echo "PPO_MAX_TOKEN_LEN_PER_GPU: $PPO_MAX_TOKEN_LEN_PER_GPU"

ray start --head

python3 -m verl.trainer.main_ppo \
    --config-name='ppo_trainer_ensemble.yaml'\
    algorithm.adv_estimator=token_reward_direct \
    data.shuffle=False \
    data.train_files="$TRAIN_DATASET" \
    data.val_files="$TEST_DATASET" \
    data.train_batch_size=64 \
    data.max_prompt_length=1024 \
    data.max_response_length=$MAX_RESP_LENGTH \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=$ACTOR_MODEL_PATH \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_activation_offload=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    $LR_ARGS \
    actor_rollout_ref.actor.ppo_mini_batch_size=$MINI_BATCH_SIZE \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$PPO_MAX_TOKEN_LEN_PER_GPU \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=$PARALLEL_SIZE \
    $KL_ARGS \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.fsdp_config.forward_prefetch=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=$PPO_MAX_TOKEN_LEN_PER_GPU \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=$TEMPERATURE \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$PARALLEL_SIZE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.75 \
    actor_rollout_ref.rollout.n=$N_RESPONSES \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    +actor_rollout_ref.rollout.val_kwargs.max_new_tokens=$MAX_VAL_RESP_LENGTH \
    actor_rollout_ref.rollout.val_kwargs.n=32 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    reward_model.enable=True \
    reward_model.model.path=/home/test/testdata/models/Qwen3-8B \
    reward_model.model.use_remove_padding=True \
    reward_model.model.fsdp_config.param_offload=False \
    reward_model.micro_batch_size_per_gpu=24 \
    custom_reward_function.path="./verl/utils/reward_score/ttrl_math/__init__.py" \
    custom_reward_function.name=reward_func \
    trainer.val_before_train=True \
    trainer.log_val_generations=2 \
    trainer.logger=['console','swanlab'] \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=20 \
    trainer.total_epochs=5 \
    trainer.default_local_dir="$CKPT_PATH"/"$PROJECT_NAME"/"$EXPERIMENT_NAME"
