# verl: Volcano Engine Reinforcement Learning for LLMs

## Project Overview

**verl** is a flexible, efficient, and production-ready distributed RL training library for large language models (LLMs), initiated by ByteDance Seed team and maintained by the verl community. It is the open-source version of the **HybridFlow** paper (EuroSys 2025).

The library enables:
- Easy extension of diverse RL algorithms (PPO, GRPO, ReMax, REINFORCE++, RLOO, DAPO, etc.)
- Seamless integration with existing LLM infrastructure (FSDP, Megatron-LM, vLLM, SGLang)
- Flexible device mapping for efficient resource utilization
- Support for vision-language models (VLMs) and multi-modal RL
- Multi-turn tool calling and agentic RL
- Scalability up to 671B models and hundreds of GPUs

## Technology Stack

### Core Dependencies
- **Python**: >= 3.10
- **Deep Learning Framework**: PyTorch with FSDP/FSDP2 or Megatron-LM
- **Distributed Computing**: Ray (>= 2.41.0) for distributed task orchestration
- **Inference Engines**: vLLM (>= 0.8.5, <= 0.11.0), SGLang (0.5.2), or Hugging Face Transformers
- **Data Management**: TensorDict (>= 0.8.0, <= 0.10.0), PyArrow (>= 19.0.0)
- **Configuration**: Hydra-core for configuration management
- **Experiment Tracking**: Weights & Biases, TensorBoard, MLflow, SwanLab

### Optional Dependencies
- **GPU Acceleration**: Flash Attention 2, Liger Kernel
- **Math/Science Tasks**: math-verify, latex2sympy2_extended, mathruler
- **Vision Models**: torchvision, qwen_vl_utils
- **LoRA Training**: PEFT
- **ModelScope**: For users in China (set `VERL_USE_MODELSCOPE=true`)

## Project Structure

```
verl/                       # Main source code
├── protocol.py             # Core DataProto class for data transfer between modules
├── base_config.py          # Base configuration class with dict-like interface
├── version/                # Version information
├── models/                 # Model implementations
│   ├── transformers/       # HF Transformers integration
│   ├── llama/             # Llama-specific implementations
│   ├── qwen2/             # Qwen2-specific implementations
│   └── mcore/             # Megatron-LM integration
├── trainer/               # Training algorithms and entry points
│   ├── config/            # Hydra configuration files (YAML)
│   ├── ppo/               # PPO trainer implementation
│   ├── main_ppo.py        # PPO entry point
│   ├── main_generation.py # Generation entry point
│   └── fsdp_sft_trainer.py # SFT trainer
├── workers/               # Worker implementations for distributed training
│   ├── fsdp_workers.py    # FSDP-based workers (actor, critic, reference)
│   ├── megatron_workers.py # Megatron-based workers
│   ├── actor/             # Actor-specific implementations
│   ├── critic/            # Critic-specific implementations
│   ├── rollout/           # Rollout/generation implementations
│   └── sharding_manager/  # Model sharding management
├── single_controller/     # Ray-based distributed controller
│   ├── ray/              # Ray-specific implementations
│   └── base/             # Base controller abstractions
├── utils/                 # Utility functions
│   ├── reward_score/     # Reward computation for various tasks
│   ├── checkpoint/       # Checkpoint saving/loading
│   └── kernel/           # Custom CUDA kernels
├── experimental/         # Experimental features
│   ├── agent_loop/       # Multi-turn agent loops
│   └── reward/           # Experimental reward managers
└── third_party/          # Third-party integrations
    ├── vllm/            # vLLM-specific patches
    └── sglang/          # SGLang-specific patches

tests/                    # Test suite
├── special_distributed/  # Multi-GPU required tests
├── special_e2e/         # End-to-end training tests
├── special_sanity/      # Quick sanity checks
├── special_npu/         # NPU-specific tests
├── trainer/             # Trainer unit tests
├── workers/             # Worker unit tests
└── *_on_cpu.py          # CPU-only tests

examples/                 # Example training scripts
├── ppo_trainer/         # PPO examples
├── grpo_trainer/        # GRPO examples
├── sft/                 # SFT examples
├── sglang_multiturn/    # Multi-turn RL examples
└── data_preprocess/     # Data preprocessing scripts

recipe/                  # Research algorithm implementations
├── dapo/               # DAPO algorithm
├── prime/              # PRIME algorithm
├── sppo/               # Self-play preference optimization
└── ...                 # Other algorithms

docs/                   # Documentation (Sphinx-based)
.github/workflows/      # CI/CD configurations
```

## Build and Installation

### Basic Installation
```bash
# Clone the repository
git clone https://github.com/volcengine/verl
cd verl

# Basic installation (CPU/GPU compatible)
pip install -e .

# With test dependencies
pip install -e .[test]

# With vLLM support
pip install -e .[test,vllm]

# With SGLang support
pip install -e .[test,sglang]

# Full installation with all optional dependencies
pip install -e .[test,vllm,sglang,gpu,math,geo]
```

### Docker Installation
Pre-built Docker images are available:
- `verl-ci-cn-beijing.cr.volces.com/verlai/verl:vllm011.dev7`
- `verl-ci-cn-beijing.cr.volces.com/verlai/verl:app-verl0.6-transformers4.56.1-sglang0.5.2-mcore0.13.0-te2.2`

See `docker/` directory for Dockerfile and build instructions.

### NPU Installation (Ascend)
Refer to `docs/ascend_tutorial/` for Huawei Ascend NPU installation and usage.

### AMD GPU Installation (ROCm)
Refer to `docs/amd_tutorial/` for AMD ROCm installation and usage.

## Code Style Guidelines

### Linting and Formatting
We use **Ruff** for linting and formatting:
- Line length: 120 characters
- Import sorting with `isort` (verl as first-party)
- Enforced rules: pycodestyle (E), Pyflakes (F), pyupgrade (UP), flake8-bugbear (B), isort (I)

### Pre-commit Hooks
Set up pre-commit hooks to ensure code quality:
```bash
pip install pre-commit
pre-commit install

# Run on staged files
pre-commit run

# Run on all files
pre-commit run --all-files

# Run specific hook
pre-commit run --all-files ruff
```

### License Headers
All Python files must include the Apache 2.0 license header:
```python
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
```

### Type Checking
MyPy is configured but currently set to `ignore_errors = true` globally. Specific modules have stricter type checking enabled:
- `verl.trainer.config.algorithm`
- `verl.trainer.ppo.core_algos`
- `verl.workers.reward_manager`

## Testing Instructions

### Test Organization
- **`tests/*/`**: Tests organized by module (e.g., `tests/trainer/` for `verl/trainer/`)
- **`tests/special_distributed/`**: Multi-GPU required tests (run with `torchrun`)
- **`tests/special_e2e/`**: End-to-end training tests
- **`tests/special_sanity/`**: Quick sanity checks for CI
- **`tests/special_npu/`**: NPU-specific tests
- **`tests/*_on_cpu.py`**: CPU-only tests

### Running Tests
```bash
# CPU-only tests (no GPU required)
pytest -s -x tests/ --ignore-glob="*test_special_*.py" --ignore-glob='*on_cpu.py' --ignore-glob='tests/special*'

# GPU unit tests
pytest -s -x --ignore-glob="*test_special_*.py" --ignore-glob='*on_cpu.py' tests/

# Distributed tests (requires 2+ GPUs)
torchrun --standalone --nnodes=1 --nproc-per-node=2 tests/workers/actor/test_special_dp_actor.py

# E2E tests
bash tests/special_e2e/run_ppo_trainer.sh
```

### CI/CD Workflows
All tests run on GitHub Actions:
- **CPU Tests**: `cpu_unit_tests.yml` - runs on every PR/push
- **GPU Tests**: `gpu_unit_tests.yml` - runs on every PR/push
- **E2E Tests**: `e2e_*.yml` - comprehensive training tests
- **vLLM Tests**: `vllm.yml` - vLLM integration tests
- **SGLang Tests**: `sgl.yml` - SGLang integration tests
- **Pre-commit**: `pre-commit.yml` - lint and format checks

## Key Architectural Concepts

### DataProto
The core data structure for passing data between modules. It wraps a `TensorDict` for tensor data and a numpy dictionary for non-tensor data:
```python
from verl import DataProto

# DataProto contains:
# - batch: TensorDict for tensor data
# - non_tensor_batch: dict[str, np.ndarray] for non-tensor data
# - meta_info: dict for metadata
```

### Workers and Controllers
- **Workers**: Ray remote classes that perform computation (actor, critic, reference, reward)
- **Controllers**: Manage worker lifecycles and communication via `single_controller`
- **Role-based**: Workers are assigned roles (ActorRollout, Critic, RefPolicy, RewardModel)

### Training Backends
- **FSDP/FSDP2**: PyTorch native sharding (recommended for most use cases)
- **Megatron-LM**: NVIDIA's large model training framework (for very large models)

### Rollout Engines
- **vLLM**: High-throughput inference engine
- **SGLang**: For multi-turn and agentic workflows
- **HF Transformers**: For compatibility (slower)

### Configuration System
Uses Hydra for hierarchical configuration:
- Base configs in `verl/trainer/config/*.yaml`
- Generated configs: `_generated_*.yaml` (auto-generated from schema)
- Override via command line: `actor_rollout_ref.actor.strategy=fsdp2`

## Running Training

### PPO Training Example
```bash
python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=gae \
    data.train_files=$DATA_PATH/train.parquet \
    data.val_files=$DATA_PATH/test.parquet \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2
```

### Key Environment Variables
- `VERL_USE_MODELSCOPE=true`: Use ModelScope Hub instead of Hugging Face
- `VERL_USE_EXTERNAL_MODULES=module1,module2`: Load external modules
- `VERL_AUTO_PADDING=true`: Enable automatic padding in DataProto

## Documentation

Documentation is built with Sphinx:
```bash
cd docs
pip install -r requirements-docs.txt
make clean
make html
python -m http.server -d _build/html/
```

Online documentation: https://verl.readthedocs.io/

## Common Development Tasks

### Adding a New Algorithm
1. Create trainer in `verl/trainer/`
2. Add config files in `verl/trainer/config/`
3. Update `scripts/generate_trainer_config.sh` if needed
4. Add tests in `tests/trainer/`
5. Add example scripts in `examples/`

### Adding a New Reward Function
1. Add scoring function in `verl/utils/reward_score/`
2. Update reward manager in `verl/workers/reward_manager/`
3. Add tests

### Adding a New Model
- For FSDP: Follow `verl/models/` structure, register in registry
- For Megatron: Implement Megatron-specific layers in `verl/models/mcore/`

## Security Considerations

- No hardcoded credentials or API keys
- Use environment variables or secure vaults for sensitive data
- All code must pass secrets scanning (configured in CI)
- Docker images are scanned for vulnerabilities

## Contributing

See `CONTRIBUTING.md` for detailed guidelines.

Key points:
- Follow the PR template
- All checks must pass (lint, test, license)
- Update documentation for user-facing changes
- Add tests for new features

## Resources

- **Paper**: [HybridFlow: A Flexible and Efficient RLHF Framework](https://arxiv.org/abs/2409.19256v2)
- **Documentation**: https://verl.readthedocs.io/
- **GitHub**: https://github.com/volcengine/verl
- **Community**: Slack, WeChat, Twitter/X

## Version

Current version: `0.7.0.dev` (stored in `verl/version/version`)

The version follows semantic versioning:
- MAJOR: Breaking changes
- MINOR: New features (backward compatible)
- PATCH: Bug fixes
- `.dev`: Development version
