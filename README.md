# Flow Actor-Critic for Offline Reinforcement Learning

Flow Actor-Critic (FAC) is a deep RL algorithm that leverages an expressive flow-matching model not only for flow-based actor design but also for conservative critic design.

## Installation

This implementation requires Python 3.9+ and is based on JAX. The main dependencies that we used can be found in `requirements.txt`. You can install the dependencies by running:
```bash
pip install -r requirements.txt
```
> For the D4RL environments, you need to set up `mujoco210`.

## Getting Datasets
You can get datasets for offline RL training with the following command:
```bash
python get_dataset.py
```

## Reproducing the results

The main implementation of FAC is in [agents/fac.py](agents/fac.py).
To train the agent, you can run a command:
```bash
python main.py
```
You can also tune some hyper-parameters. Here are some example commands:
```bash
# OGBench puzzle-3x3-play-singletask-v0
python main.py --env_name=puzzle-3x3-play-singletask-v0 --agent.fac_alpha=1.0 --agent.fac_lambda=1.0 --agent.q_agg='mean'
# D4RL antmaze-large-play-v2
python main.py --env_name=antmaze-large-play-v2 --agent.fac_alpha=5.0 --agent.fac_lambda=0.03 --agent.fac_threshold='dataset_wide_constant'
```

### The main results of FAC

We provide the complete list of the **command-lines** to reproduce the main results of FAC in the paper.

<details>
<summary><b>Click to see the full list of commands</b></summary>

#### FAC on default tasks of OGBench (state-based)

```bash
# OGBench antmaze-large-navigate-singletask-v0 (default: task1)
python main.py --env_name=antmaze-large-navigate-singletask-v0 --agent.fac_alpha=0.5 --agent.fac_lambda=0.1 --agent.fac_threshold='dataset_wide_constant'
# OGBench antmaze-giant-navigate-singletask-v0 (default: task1)
python main.py --env_name=antmaze-giant-navigate-singletask-v0 --agent.fac_alpha=1.0 --agent.fac_lambda=0.1 --agent.fac_threshold='dataset_wide_constant'
# OGBench humanoidmaze-medium-navigate-singletask-v0 (default: task1)
python main.py --env_name=humanoidmaze-medium-navigate-singletask-v0 --agent.fac_alpha=0.5 --agent.fac_lambda=0.3 --agent.q_agg='mean' --agent.logp_method='hutch-rade'
# OGBench humanoidmaze-large-navigate-singletask-v0 (default: task1)
python main.py --env_name=humanoidmaze-large-navigate-singletask-v0 --agent.fac_alpha=0.5 --agent.fac_lambda=0.1 --agent.q_agg='mean' --agent.logp_method='hutch-rade'
# OGBench antsoccer-arena-navigate-singletask-v0 (default: task4)
python main.py --env_name=antsoccer-arena-navigate-singletask-v0 --agent.fac_alpha=1.0 --agent.fac_lambda=0.1 --agent.q_agg='mean' --agent.fac_threshold='dataset_wide_constant'
# OGBench cube-single-play-singletask-v0 (default: task2)
python main.py --env_name=cube-single-play-singletask-v0 --agent.fac_alpha=0.5 --agent.fac_lambda=10.0 --agent.q_agg='mean'
# OGBench cube-double-play-singletask-v0 (default: task2)
python main.py --env_name=cube-double-play-singletask-v0 --agent.fac_alpha=1.0 --agent.fac_lambda=1.0 --agent.q_agg='mean'
# OGBench scene-play-singletask-v0 (default: task2)
python main.py --env_name=scene-play-singletask-v0 --agent.fac_alpha=1.0 --agent.fac_lambda=1.0 --agent.q_agg='mean'
# OGBench puzzle-3x3-play-singletask-v0 (default: task4)
python main.py --env_name=puzzle-3x3-play-singletask-v0 --agent.fac_alpha=1.0 --agent.fac_lambda=1.0 --agent.q_agg='mean'
# OGBench puzzle-4x4-play-singletask-v0 (default: task4)
python main.py --env_name=puzzle-4x4-play-singletask-v0 --agent.fac_alpha=5.0 --agent.fac_lambda=0.3 --agent.q_agg='mean'
```

#### FAC on all tasks of OGBench (state-based)

```bash
# OGBench antmaze-large-navigate-singletask-{task1, task2, task3, task4, task5}-v0 (default: task1)
python main.py --env_name=antmaze-large-navigate-singletask-task1-v0 --agent.fac_alpha=0.5 --agent.fac_lambda=0.1 --agent.fac_threshold='dataset_wide_constant'
python main.py --env_name=antmaze-large-navigate-singletask-task2-v0 --agent.fac_alpha=0.5 --agent.fac_lambda=0.1 --agent.fac_threshold='dataset_wide_constant'
python main.py --env_name=antmaze-large-navigate-singletask-task3-v0 --agent.fac_alpha=0.5 --agent.fac_lambda=0.1 --agent.fac_threshold='dataset_wide_constant'
python main.py --env_name=antmaze-large-navigate-singletask-task4-v0 --agent.fac_alpha=0.5 --agent.fac_lambda=0.1 --agent.fac_threshold='dataset_wide_constant'
python main.py --env_name=antmaze-large-navigate-singletask-task5-v0 --agent.fac_alpha=0.5 --agent.fac_lambda=0.1 --agent.fac_threshold='dataset_wide_constant'
# OGBench antmaze-giant-navigate-singletask-{task1, task2, task3, task4, task5}-v0 (default: task1)
python main.py --env_name=antmaze-giant-navigate-singletask-task1-v0 --agent.fac_alpha=1.0 --agent.fac_lambda=0.1 --agent.fac_threshold='dataset_wide_constant'
python main.py --env_name=antmaze-giant-navigate-singletask-task2-v0 --agent.fac_alpha=1.0 --agent.fac_lambda=0.1 --agent.fac_threshold='dataset_wide_constant'
python main.py --env_name=antmaze-giant-navigate-singletask-task3-v0 --agent.fac_alpha=1.0 --agent.fac_lambda=0.1 --agent.fac_threshold='dataset_wide_constant'
python main.py --env_name=antmaze-giant-navigate-singletask-task4-v0 --agent.fac_alpha=1.0 --agent.fac_lambda=0.1 --agent.fac_threshold='dataset_wide_constant'
python main.py --env_name=antmaze-giant-navigate-singletask-task5-v0 --agent.fac_alpha=1.0 --agent.fac_lambda=0.1 --agent.fac_threshold='dataset_wide_constant'
# OGBench humanoidmaze-medium-navigate-singletask-{task1, task2, task3, task4, task5}-v0 (default: task1)
python main.py --env_name=humanoidmaze-medium-navigate-singletask-task1-v0 --agent.fac_alpha=0.5 --agent.fac_lambda=0.3 --agent.q_agg='mean' --agent.logp_method='hutch-rade'
python main.py --env_name=humanoidmaze-medium-navigate-singletask-task2-v0 --agent.fac_alpha=0.5 --agent.fac_lambda=0.3 --agent.q_agg='mean' --agent.logp_method='hutch-rade'
python main.py --env_name=humanoidmaze-medium-navigate-singletask-task3-v0 --agent.fac_alpha=0.5 --agent.fac_lambda=0.3 --agent.q_agg='mean' --agent.logp_method='hutch-rade'
python main.py --env_name=humanoidmaze-medium-navigate-singletask-task4-v0 --agent.fac_alpha=0.5 --agent.fac_lambda=0.3 --agent.q_agg='mean' --agent.logp_method='hutch-rade'
python main.py --env_name=humanoidmaze-medium-navigate-singletask-task5-v0 --agent.fac_alpha=0.5 --agent.fac_lambda=0.3 --agent.q_agg='mean' --agent.logp_method='hutch-rade'
# OGBench humanoidmaze-large-navigate-singletask-{task1, task2, task3, task4, task5}-v0 (default: task1)
python main.py --env_name=humanoidmaze-large-navigate-singletask-task1-v0 --agent.fac_alpha=0.5 --agent.fac_lambda=0.1 --agent.q_agg='mean' --agent.logp_method='hutch-rade'
python main.py --env_name=humanoidmaze-large-navigate-singletask-task2-v0 --agent.fac_alpha=0.5 --agent.fac_lambda=0.1 --agent.q_agg='mean' --agent.logp_method='hutch-rade'
python main.py --env_name=humanoidmaze-large-navigate-singletask-task3-v0 --agent.fac_alpha=0.5 --agent.fac_lambda=0.1 --agent.q_agg='mean' --agent.logp_method='hutch-rade'
python main.py --env_name=humanoidmaze-large-navigate-singletask-task4-v0 --agent.fac_alpha=0.5 --agent.fac_lambda=0.1 --agent.q_agg='mean' --agent.logp_method='hutch-rade'
python main.py --env_name=humanoidmaze-large-navigate-singletask-task5-v0 --agent.fac_alpha=0.5 --agent.fac_lambda=0.1 --agent.q_agg='mean' --agent.logp_method='hutch-rade'
# OGBench antsoccer-arena-navigate-singletask-{task1, task2, task3, task4, task5}-v0 (default: task4)
python main.py --env_name=antsoccer-arena-navigate-singletask-task1-v0 --agent.fac_alpha=1.0 --agent.fac_lambda=0.1 --agent.q_agg='mean' --agent.fac_threshold='dataset_wide_constant'
python main.py --env_name=antsoccer-arena-navigate-singletask-task2-v0 --agent.fac_alpha=1.0 --agent.fac_lambda=0.1 --agent.q_agg='mean' --agent.fac_threshold='dataset_wide_constant'
python main.py --env_name=antsoccer-arena-navigate-singletask-task3-v0 --agent.fac_alpha=1.0 --agent.fac_lambda=0.1 --agent.q_agg='mean' --agent.fac_threshold='dataset_wide_constant'
python main.py --env_name=antsoccer-arena-navigate-singletask-task4-v0 --agent.fac_alpha=1.0 --agent.fac_lambda=0.1 --agent.q_agg='mean' --agent.fac_threshold='dataset_wide_constant'
python main.py --env_name=antsoccer-arena-navigate-singletask-task5-v0 --agent.fac_alpha=1.0 --agent.fac_lambda=0.1 --agent.q_agg='mean' --agent.fac_threshold='dataset_wide_constant'
# OGBench cube-single-play-singletask-{task1, task2, task3, task4, task5}-v0 (default: task2)
python main.py --env_name=cube-single-play-singletask-task1-v0 --agent.fac_alpha=0.5 --agent.fac_lambda=10.0 --agent.q_agg='mean'
python main.py --env_name=cube-single-play-singletask-task2-v0 --agent.fac_alpha=0.5 --agent.fac_lambda=10.0 --agent.q_agg='mean'
python main.py --env_name=cube-single-play-singletask-task3-v0 --agent.fac_alpha=0.5 --agent.fac_lambda=10.0 --agent.q_agg='mean'
python main.py --env_name=cube-single-play-singletask-task4-v0 --agent.fac_alpha=0.5 --agent.fac_lambda=10.0 --agent.q_agg='mean'
python main.py --env_name=cube-single-play-singletask-task5-v0 --agent.fac_alpha=0.5 --agent.fac_lambda=10.0 --agent.q_agg='mean'
# OGBench cube-double-play-singletask-{task1, task2, task3, task4, task5}-v0 (default: task2)
python main.py --env_name=cube-double-play-singletask-task1-v0 --agent.fac_alpha=1.0 --agent.fac_lambda=1.0 --agent.q_agg='mean'
python main.py --env_name=cube-double-play-singletask-task2-v0 --agent.fac_alpha=1.0 --agent.fac_lambda=1.0 --agent.q_agg='mean'
python main.py --env_name=cube-double-play-singletask-task3-v0 --agent.fac_alpha=1.0 --agent.fac_lambda=1.0 --agent.q_agg='mean'
python main.py --env_name=cube-double-play-singletask-task4-v0 --agent.fac_alpha=1.0 --agent.fac_lambda=1.0 --agent.q_agg='mean'
python main.py --env_name=cube-double-play-singletask-task5-v0 --agent.fac_alpha=1.0 --agent.fac_lambda=1.0 --agent.q_agg='mean'
# OGBench scene-play-singletask-{task1, task2, task3, task4, task5}-v0 (default: task2)
python main.py --env_name=scene-play-singletask-task1-v0 --agent.fac_alpha=1.0 --agent.fac_lambda=1.0 --agent.q_agg='mean'
python main.py --env_name=scene-play-singletask-task2-v0 --agent.fac_alpha=1.0 --agent.fac_lambda=1.0 --agent.q_agg='mean'
python main.py --env_name=scene-play-singletask-task3-v0 --agent.fac_alpha=1.0 --agent.fac_lambda=1.0 --agent.q_agg='mean'
python main.py --env_name=scene-play-singletask-task4-v0 --agent.fac_alpha=1.0 --agent.fac_lambda=1.0 --agent.q_agg='mean'
python main.py --env_name=scene-play-singletask-task5-v0 --agent.fac_alpha=1.0 --agent.fac_lambda=1.0 --agent.q_agg='mean'
# OGBench puzzle-3x3-play-singletask-{task1, task2, task3, task4, task5}-v0 (default: task4)
python main.py --env_name=puzzle-3x3-play-singletask-task1-v0 --agent.fac_alpha=1.0 --agent.fac_lambda=1.0 --agent.q_agg='mean'
python main.py --env_name=puzzle-3x3-play-singletask-task2-v0 --agent.fac_alpha=1.0 --agent.fac_lambda=1.0 --agent.q_agg='mean'
python main.py --env_name=puzzle-3x3-play-singletask-task3-v0 --agent.fac_alpha=1.0 --agent.fac_lambda=1.0 --agent.q_agg='mean'
python main.py --env_name=puzzle-3x3-play-singletask-task4-v0 --agent.fac_alpha=1.0 --agent.fac_lambda=1.0 --agent.q_agg='mean'
python main.py --env_name=puzzle-3x3-play-singletask-task5-v0 --agent.fac_alpha=1.0 --agent.fac_lambda=1.0 --agent.q_agg='mean'
# OGBench puzzle-4x4-play-singletask-{task1, task2, task3, task4, task5}-v0 (default: task4)
python main.py --env_name=puzzle-4x4-play-singletask-task1-v0 --agent.fac_alpha=5.0 --agent.fac_lambda=0.3 --agent.q_agg='mean'
python main.py --env_name=puzzle-4x4-play-singletask-task2-v0 --agent.fac_alpha=5.0 --agent.fac_lambda=0.3 --agent.q_agg='mean'
python main.py --env_name=puzzle-4x4-play-singletask-task3-v0 --agent.fac_alpha=5.0 --agent.fac_lambda=0.3 --agent.q_agg='mean'
python main.py --env_name=puzzle-4x4-play-singletask-task4-v0 --agent.fac_alpha=5.0 --agent.fac_lambda=0.3 --agent.q_agg='mean'
python main.py --env_name=puzzle-4x4-play-singletask-task5-v0 --agent.fac_alpha=5.0 --agent.fac_lambda=0.3 --agent.q_agg='mean'
```

#### FAC on visual tasks of OGBench (pixel-based)

```bash
# OGBench visual-cube-single-play-singletask-task1-v0
python main.py --env_name=visual-cube-single-play-singletask-task1-v0 --agent.fac_alpha=0.5 --agent.fac_lambda=10.0 --agent.q_agg='mean' --offline_steps=500000 --agent.encoder=impala_small --p_aug=0.5 --frame_stack=3
# OGBench visual-cube-double-play-singletask-task1-v0
python main.py --env_name=visual-cube-double-play-singletask-task1-v0 --agent.fac_alpha=1.0 --agent.fac_lambda=1.0 --agent.q_agg='mean' --offline_steps=500000 --agent.encoder=impala_small --p_aug=0.5 --frame_stack=3
# OGBench visual-scene-play-singletask-task1-v0
python main.py --env_name=visual-scene-play-singletask-task1-v0 --agent.fac_alpha=1.0 --agent.fac_lambda=1.0 --agent.q_agg='mean' --offline_steps=500000 --agent.encoder=impala_small --p_aug=0.5 --frame_stack=3
# OGBench visual-puzzle-3x3-play-singletask-task1-v0
python main.py --env_name=visual-puzzle-3x3-play-singletask-task1-v0 --agent.fac_alpha=1.0 --agent.fac_lambda=1.0 --agent.q_agg='mean' --offline_steps=500000 --agent.encoder=impala_small --p_aug=0.5 --frame_stack=3
# OGBench visual-puzzle-4x4-play-singletask-task1-v0
python main.py --env_name=visual-puzzle-4x4-play-singletask-task1-v0 --agent.fac_alpha=5.0 --agent.fac_lambda=0.3 --agent.q_agg='mean' --offline_steps=500000 --agent.encoder=impala_small --p_aug=0.5 --frame_stack=3
```

#### FAC on D4RL

```bash
# D4RL halfcheetah-medium-v2
python main.py --env_name=halfcheetah-medium-v2 --agent.fac_alpha=0.05 --agent.fac_lambda=0.0003
# D4RL halfcheetah-medium-replay-v2
python main.py --env_name=halfcheetah-medium-replay-v2 --agent.fac_alpha=0.05 --agent.fac_lambda=0.0003
# D4RL halfcheetah-medium-expert-v2
python main.py --env_name=halfcheetah-medium-expert-v2 --agent.fac_alpha=0.5 --agent.fac_lambda=0.003
# D4RL hopper-medium-v2
python main.py --env_name=hopper-medium-v2 --agent.fac_alpha=5.0 --agent.fac_lambda=0.1
# D4RL hopper-medium-replay-v2
python main.py --env_name=hopper-medium-replay-v2 --agent.fac_alpha=5.0 --agent.fac_lambda=0.1
# D4RL hopper-medium-expert-v2
python main.py --env_name=hopper-medium-expert-v2 --agent.fac_alpha=5.0 --agent.fac_lambda=0.3
# D4RL walker2d-medium-v2
python main.py --env_name=walker2d-medium-v2 --agent.fac_alpha=5.0 --agent.fac_lambda=0.03
# D4RL walker2d-medium-replay-v2
python main.py --env_name=walker2d-medium-replay-v2 --agent.fac_alpha=5.0 --agent.fac_lambda=0.1
# D4RL walker2d-medium-expert-v2
python main.py --env_name=walker2d-medium-expert-v2 --agent.fac_alpha=5.0 --agent.fac_lambda=0.03

# D4RL antmaze-umaze-v2
python main.py --env_name=antmaze-umaze-v2 --agent.fac_alpha=1.0 --agent.fac_lambda=0.1 --agent.fac_threshold='dataset_wide_constant'
# D4RL antmaze-umaze-diverse-v2
python main.py --env_name=antmaze-umaze-diverse-v2 --agent.fac_alpha=1.0 --agent.fac_lambda=0.1 --agent.fac_threshold='dataset_wide_constant'
# D4RL antmaze-medium-play-v2
python main.py --env_name=antmaze-medium-play-v2 --agent.fac_alpha=0.5 --agent.fac_lambda=0.03 --agent.fac_threshold='dataset_wide_constant'
# D4RL antmaze-medium-diverse-v2
python main.py --env_name=antmaze-medium-diverse-v2 --agent.fac_alpha=5.0 --agent.fac_lambda=0.1 --agent.fac_threshold='dataset_wide_constant'
# D4RL antmaze-large-play-v2
python main.py --env_name=antmaze-large-play-v2 --agent.fac_alpha=5.0 --agent.fac_lambda=0.03 --agent.fac_threshold='dataset_wide_constant'
# D4RL antmaze-large-diverse-v2
python main.py --env_name=antmaze-large-diverse-v2 --agent.fac_alpha=1.0 --agent.fac_lambda=0.03 --agent.fac_threshold='dataset_wide_constant'

# D4RL pen-human-v1
python main.py --env_name=pen-human-v1 --agent.fac_alpha=10.0 --agent.fac_lambda=0.3 --agent.logp_method='hutch-rade'
# D4RL pen-cloned-v1
python main.py --env_name=pen-cloned-v1 --agent.fac_alpha=1.0 --agent.fac_lambda=0.1 --agent.logp_method='hutch-rade'
# D4RL door-human-v1
python main.py --env_name=door-human-v1 --agent.fac_alpha=1.0 --agent.fac_lambda=3.0 --agent.logp_method='hutch-rade'
# D4RL door-cloned-v1
python main.py --env_name=door-cloned-v1 --agent.fac_alpha=5.0 --agent.fac_lambda=10.0 --agent.logp_method='hutch-rade'
# D4RL hammer-human-v1
python main.py --env_name=hammer-human-v1 --agent.fac_alpha=0.5 --agent.fac_lambda=0.03 --agent.logp_method='hutch-rade'
# D4RL hammer-cloned-v1
python main.py --env_name=hammer-cloned-v1 --agent.fac_alpha=0.5 --agent.fac_lambda=1.0 --agent.logp_method='hutch-rade'
# D4RL relocate-human-v1
python main.py --env_name=relocate-human-v1 --agent.fac_alpha=5.0 --agent.fac_lambda=10.0 --agent.logp_method='hutch-rade'
# D4RL relocate-cloned-v1
python main.py --env_name=relocate-cloned-v1 --agent.fac_alpha=5.0 --agent.fac_lambda=10.0 --agent.logp_method='hutch-rade'
```
</details>

### The main results of baseline

We also provide the complete list of the **command-lines** to reproduce the main results of the baseline in the paper.

<details>
<summary><b>Click to see the full list of commands</b></summary>

#### Baseline: FQL on MuJoCo tasks of D4RL
```bash
# D4RL halfcheetah-medium-v2
python main_baselines.py --agent='agents/fql.py' --env_name=halfcheetah-medium-v2 --agent.alpha=3.0
# D4RL halfcheetah-medium-replay-v2
python main_baselines.py --agent='agents/fql.py' --env_name=halfcheetah-medium-replay-v2 --agent.alpha=30.0
# D4RL halfcheetah-medium-expert-v2
python main_baselines.py --agent='agents/fql.py' --env_name=halfcheetah-medium-expert-v2 --agent.alpha=30.0
# D4RL hopper-medium-v2
python main_baselines.py --agent='agents/fql.py' --env_name=hopper-medium-v2 --agent.alpha=100.0
# D4RL hopper-medium-replay-v2
python main_baselines.py --agent='agents/fql.py' --env_name=hopper-medium-replay-v2 --agent.alpha=100.0
# D4RL hopper-medium-expert-v2
python main_baselines.py --agent='agents/fql.py' --env_name=hopper-medium-expert-v2 --agent.alpha=100.0
# D4RL walker2d-medium-v2
python main_baselines.py --agent='agents/fql.py' --env_name=walker2d-medium-v2 --agent.alpha=300.0
# D4RL walker2d-medium-replay-v2
python main_baselines.py --agent='agents/fql.py' --env_name=walker2d-medium-replay-v2 --agent.alpha=300.0
# D4RL walker2d-medium-expert-v2
python main_baselines.py --agent='agents/fql.py' --env_name=walker2d-medium-expert-v2 --agent.alpha=300.0
```
</details>


## Base Implementation

This codebase was built upon the implementation of [FQL](https://github.com/seohongpark/fql), and with reference to [Flow Matching](https://github.com/facebookresearch/flow_matching).
