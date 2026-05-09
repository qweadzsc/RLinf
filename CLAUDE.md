# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RLinf is a flexible and scalable open-source reinforcement learning infrastructure for Embodied and Agentic AI. It supports multiple training paradigms:
- **Embodied AI**: Robotics environments (ManiSkill, LIBERO, RoboCasa, IsaacLab, Franka, etc.)
- **Agentic AI**: Single-agent (Search-R1, rStar2) and multi-agent (WideSeek-R1) systems
- **Reasoning**: Math reasoning RL with GRPO/PPO algorithms

Key architectural feature: **Macro-to-Micro Flow Transformation** - decouples logical workflow from physical execution, enabling flexible distributed scheduling across GPU clusters.

## Common Commands

### Running Tests

**Unit tests:**
```bash
cd tests/unit_tests
pytest test_worker.py -v
```

**E2E tests (require GPU cluster):**
```bash
# Reasoning
REPO_PATH=$(pwd) tests/e2e_tests/reasoning/run.sh qwen2.5-1.5b-grpo-collocated-mg-sgl

# Agent (Search-R1)
tests/e2e_tests/agent/searchr1/run-train.sh

# Embodied
tests/e2e_tests/embodied/run.sh maniskill_ppo_openvla
```

### Running Examples

**Math reasoning with GRPO:**
```bash
cd examples/reasoning
./run_main_grpo_math.sh [config_name]
```

**Search-R1 agent training:**
```bash
cd examples/agent/searchr1
./run_train.sh [config_name]
```

**Embodied RL:**
```bash
cd examples/embodiment
./run_embodiment.sh [config_name]
```

### Environment Setup

```bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TOKENIZERS_PARALLELISM=false
export RAY_DEDUP_LOGS=0
```

## Core Architecture

### Workers (Distributed Components)

RLinf decomposes RL training into **Workers** (distributed actors) that communicate via **Channels**:

- **Rollout Worker**: Generates trajectories using model inference (SGLang/vLLM backend)
- **Actor Worker**: Updates policy model parameters (FSDP/Megatron backend)
- **Reward Worker**: Computes rewards from environment interactions
- **Critic Worker**: (Optional) Value function for PPO
- **AgentLoop Worker**: (Agentic AI) Manages multi-turn tool interactions
- **Tool Worker**: (Agentic AI) Executes external tool calls (search, code execution)

### Placement Strategies

Three placement modes for mapping workers to GPU nodes:

- **`all` (collocated)**: All workers on same node via pipeline parallelism
- **`separate` (disaggregated)**: Each worker type on dedicated nodes for scale-out
- **`auto` (dynamic)**: Scheduler dynamically reassigns resources mid-training

Configuration via `cluster.component_placement` in YAML configs:

```yaml
cluster:
  component_placement:
    actor,rollout,reward: all  # π₀ example
    actor,rollout,reward: separate  # Scale-out
    actor,rollout,reward: auto  # Dynamic scheduling
```

### Runners

Orchestrate worker lifecycle and training loop:

- **`ReasoningRunner`**: Standard reasoning RL (PPO, GRPO) - single-turn generation
- **`EmbodiedRunner`**: Embodied RL with simulator environments
- **`AgentRunner`**: Agentic systems with tool interaction (Search-R1, WideSeek-R1)
- **`SACRunner`**: Q-learning based algorithms for continuous control

### Backends

**Model Training:**
- **FSDP**: Full Sharded Data Parallel (HuggingFace models) - faster prototyping
- **Megatron-LM**: Tensor/pipeline parallelism - optimized for large-scale

**Inference:**
- **SGLang**: RadixAttention, CUDA graph - recommended for agentic/reasoning
- **vLLM**: PagedAttention - alternative inference backend

Configure via `rollout.rollout_backend: sglang|vllm` and actor backend in configs.

## Configuration System

All training defined by Hydra YAML configs (`--config-path`, `--config-name`). See `examples/*/config/`.

**Key sections:**
- `cluster`: Distributed topology (nodes, component placement)
- `algorithm`: RL hyperparameters (group_size, learning rate, KL penalty)
- `rollout`: Inference backend (SGLang/vLLM settings, batch size)
- `actor`: Training backend (FSDP/Megatron settings, optimizer)
- `data`: Dataset path, prompt format, max length
- `agentloop`: (Agentic only) Tool configs, max turns

## Important Directories

- `rlinf/workers/`: Worker implementations (actors, rollout, reward, agent_loop)
- `rlinf/scheduler/`: Distributed coordination (placement, channels, clusters)
- `rlinf/algorithms/`: RL algorithms (GRPO, PPO, SAC loss computation)
- `rlinf/envs/`: Embodied environment integrations (ManiSkill, LIBERO, IsaacLab, etc.)
- `rlinf/agents/`: Agent-specific implementations (searchr1, wideseek_r1)
- `examples/`: Ready-to-run training scripts and configs
- `tests/`: E2E tests for each supported scenario

## Key Implementation Patterns

**Session Management (rlinf/workers/rollout/rollout_worker.py:600-620):**
Workers communicate via Channel session keys to ensure causality (e.g., rollout outputs -> actor inputs).

**Dynamic Rebalancing (rlinf/scheduler/dynamic_scheduler/):**
Auto mode fetches profiler metrics and redeploys workers mid-training for load balancing.

**Tool Integration (rlinf/workers/agent/agent_loop.py):**
AgentLoop uses toolcall_parser to extract `<tool_name>(args)` from LLM output, dispatches to ToolWorker, injects response back into prompt for next generation turn.

**Reward Shaping (rlinf/workers/reward/reward_worker.py):**
RewardWorker applies task-specific transformations (e.g., math answer correctness, embodied success criteria, agent tool usage penalty).

## Notes for Code Changes

- **Hydra config changes**: Always update `rlinf/config.py` to declare new config fields with types and defaults
- **New worker type**: Inherit from `BaseWorker`, implement `create_group()`, add to `rlinf/scheduler/scheduler_worker.py` component registry
- **Custom environments**: Add environment class to `rlinf/envs/`, register in `rlinf/envs/__init__.py`
- **Custom models**: Use `@register_model` decorator, add to `rlinf/models/model_registry.py`
- **New RL algorithm**: Implement loss function in `rlinf/algorithms/`, register in runner

## Testing Philosophy

- **Unit tests** (`tests/unit_tests/`): Test worker logic in isolation without GPU
- **E2E tests** (`tests/e2e_tests/`): Full training run on small configs to verify integration
- Use `--max_steps 2` in configs for quick smoke tests before full runs