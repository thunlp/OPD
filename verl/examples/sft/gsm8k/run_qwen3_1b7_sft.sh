set -x

if [ "$#" -lt 2 ]; then
    echo "Usage: run_qwen3_1b7_sft.sh <nproc_per_node> <save_path> [other_configs...]"
    exit 1
fi

nproc_per_node=$1
save_path=$2

# Shift the arguments so $@ refers to the rest
shift 2

torchrun --standalone --nnodes=1 --nproc_per_node=$nproc_per_node \
     -m verl.trainer.fsdp_sft_trainer \
    data.train_files=datasets/DeepMath-103K/verl_format/train.parquet \
    data.val_files="[datasets/test_data/AIME25/test.parquet,datasets/test_data/AMC23/test.parquet]" \
    data.max_length=16384 \
    data.micro_batch_size_per_gpu=4 \
    model.partial_pretrain=Qwen/Qwen3-1.8B-Instruct \
    trainer.default_local_dir=$save_path \
    trainer.project_name=deepmath-sft \
    trainer.experiment_name=deepmath-sft-qwen3-1b7 \
    trainer.total_epochs=4 \
    trainer.logger='["console","wandb"]' $@


