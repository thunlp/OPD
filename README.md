<div align="center">

# Rethinking On-Policy Distillation of Large Language Models:<br>Phenomenology, Mechanism, and Recipe

[![Paper](https://img.shields.io/badge/paper-A42C25?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/abs/xxxx.xxxxx)  [![Github](https://img.shields.io/badge/OPD-000000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/thunlp/OPD)

</div>

<div align="center" style="font-family: Arial, sans-serif;">
  <p>
    <a href="#news" style="text-decoration: none; font-weight: bold;">🎉 News</a> •
    <a href="#overview" style="text-decoration: none; font-weight: bold;">📖 Overview</a> •
    <a href="#getting-started" style="text-decoration: none; font-weight: bold;">✨ Getting Started</a>
  </p>
  <p>
    <a href="#contact" style="text-decoration: none; font-weight: bold;">📨 Contact</a> •
    <a href="#citation" style="text-decoration: none; font-weight: bold;">🎈 Citation</a>
  </p>
</div>

---

## News

- **[2026-04-15]** We investigate the dynamics and mechanisms of on-policy distillation (OPD) of LLMs, and propose practical strategies to recover failing OPD. Check it out: [Paper](https://arxiv.org/abs/xxxx.xxxxx).

## Overview

![1776212644959](figs/opd_teaser.png)

On-policy distillation (OPD) has become a core technique in the post-training of large language models, yet its training dynamics remain poorly understood.
This paper provides a systematic investigation of OPD dynamics and mechanisms.
We first identify that two conditions govern whether OPD succeeds or fails: (i) the student and teacher should share compatible thinking patterns; and (ii) even with consistent thinking patterns and higher scores, the teacher must offer genuinely new capabilities beyond what the student has seen during training.
We validate these findings through weak-to-strong reverse distillation, showing that same-family 1.5B and 7B teachers are distributionally indistinguishable from the student’s perspective.
Probing into the token-level mechanism, we show that successful OPD is characterized by progressive alignment on high-probability tokens at student-visited states, a small shared token set that concentrates most of the probability mass (97\%--99\%).
We further propose two practical strategies to recover failing OPD: off-policy cold start and teacher-aligned prompt selection.
Finally, we show that OPD's apparent free lunch of dense token-level reward comes at a cost, raising the question of whether OPD can scale to long-horizon distillation.

## Getting Started

### Environment Setup

Our code is mainly based on [verl](https://github.com/verl-project/verl) (v0.7.0). To prepare the environment used for OPD and RL:

```bash
conda create -n verl python==3.12
conda activate verl
cd verl/
USE_MEGATRON=0 bash scripts/install_vllm_sglang_mcore.sh
pip install math-verify
```

And we use [LlamaFactory](https://github.com/hiyouga/LLaMA-Factory) (v0.9.5) for SFT training:

```bash
conda create -n sft python==3.11
cd LlamaFactory/
pip install -e .
pip install -r requirements/metrics.txt
```

### Training

#### OPD

Use the following command to start on-policy distillation:

```bash
bash on_policy_distillation.sh
```

<details>
<summary><b>Key Parameters</b></summary>

| Parameter | Default | Description |
|-----------|---------|-------------|
| **Distillation Method** |||
| `ADV_ESTIMATOR` | `token_reward_direct` | It can't be modified if you use OPD |
| `ACTOR_MODEL_PATH` | — | Path to the student (policy) model to be trained |
| `REWARD_MODEL_PATH` | — | Path to the teacher model that provides token-level reward signals |
| **Generation Control** |||
| `N_RESPONSES` | `4` | Number of rollout responses generated per prompt |
| `MAX_PROMPT_LENGTH` | `1024` | Maximum token length for prompts |
| `MAX_RESP_LENGTH` | `7168` | Maximum token length for responses during training |
| `MAX_VAL_RESP_LENGTH` | `31744` | Maximum token length for responses during validation (set larger to ensure complete generation) |
| **Top-K & Weighting Strategy** |||
| `LOG_PROB_TOP_K` | `16` | Number of Top-K tokens retained when computing token-level rewards; setting to `0` falls back to sampled-token OPD |
| `TOP_K_STRATEGY` | `only_stu` | Strategy for selecting the Top-K token set. Options: `only_stu` (select Top-K from the student, then query the teacher for corresponding log-probs), `only_tch` (select Top-K from the teacher), `intersection` (keep tokens appearing in both student and teacher Top-K), `union` (merge student and teacher Top-K), `union-intersection` (tokens in either Top-K but not both, i.e. symmetric difference) |
| `REWARD_WEIGHT_MODE` | `student_p` | Weighting scheme for token rewards. `student_p`: weighted by student probability; `teacher_p`: weighted by teacher probability; `none`: no weighting |

</details>

> [!NOTE]
> You can use `scripts/infer/dedup_deepmath.py` to deduplicate DeepMath against DAPO-Math-17K and avoid data overlap, as the experiments shown in Section 5.2 in our paper.

#### SFT

Use `scripts/infer/vllm_rollout.py` to rollout teacher responses that will later be used for student SFT.

<details>
<summary><b>Key Parameters</b></summary>

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--input-parquet` | required | Path to the parquet file that provides prompts for teacher rollout |
| `--model-path` | required | Path to the teacher model checkpoint used to generate responses |
| `--gpu-ids` | `0,1,2,3,4,5,6,7` | Comma-separated GPU IDs used for multiprocessing rollout |
| `--enable-thinking` | `false` | Whether to enable the model's thinking template when formatting prompts |
| `--enable-rejection-sampling` | `true` | Whether to reject invalid outputs and retry generation |
| `--max-attempts-per-rollout` | `3` | Maximum number of retries for each rollout slot when rejection sampling is enabled |

</details>

Below is an example command for generating teacher responses with `Qwen3-4B (Non-thinking)`:

```bash
python scripts/infer/vllm_rollout.py \
  --input-parquet datasets/OpenThoughts3-1.2M-math.parquet \
  --model-path model/Qwen3-4B \
  --gpu-ids 0,1,2,3,4,5,6,7 \
  --enable-thinking false \
  --enable-rejection-sampling true \
  --max-attempts-per-rollout 3
```

After the rollout finishes, use the generated teacher responses for student SFT. An example SFT training command is:

```bash
llamafactory-cli train LlamaFactory/examples/train_full/qwen3_base_full_sft.yaml
```

#### RL (GRPO)

We use GRPO as the RL algorithm. To enable RL, set `ADV_ESTIMATOR=grpo` and `LOG_PROB_TOP_K=0`. A reference script `grpo.sh` is provided.

> [!IMPORTANT]
> **Non-thinking Models:** When training a non-thinking model (e.g., `Qwen3-1.7B (Non-thinking)`) using OPD or RL, you must add `+data.apply_chat_template_kwargs.enable_thinking=False` to the training script.

### Validation

We reuse the evaluation pipeline from [JustRL](https://github.com/thunlp/JustRL).

**Generation (Optional)**

```bash
cd scripts/val/eval
python gen_vllm.py
```

Before running generation, set `MODEL_NAMES` in `gen_vllm.py` to the checkpoint(s) you want to evaluate. And set appropriate `available_workers`.

**Grading**

```bash
cd scripts/val/eval
python grade.py
```

The grading script processes all JSONL files in the output directory and generates grading_results.json. If needed, you can enable the LLM-based verifier with:

```bash
python grade.py --enable_model_verifier
```

*All experiments were conducted on 8 x NVIDIA A800 80GB GPUs.*

# 📨Contact

- Bingxiang He: hebx24@mails.tsinghua.edu.cn
- Ning Ding: dingning@mail.tsinghua.edu.cn

## Citation

If you find this work helpful, please cite us:

```bibtex

```
