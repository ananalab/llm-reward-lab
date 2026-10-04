# llm-reward-lab

Small Eureka-style loop: an LLM writes reward functions (JAX) for MuJoCo Playground tasks, PPO (Brax) trains a policy on each one, and the LLM gets feedback before writing the next batch.

The question I want to answer: does the LLM fix its rewards better (and spot reward hacking more often) when it sees frames of what the robot actually does, on top of the usual training statistics?

Three conditions, same LLM, same number of trained candidates:

- **A**: no feedback, K x N samples of the initial prompt
- **B**: text feedback (task score, per-component curves, reward/score correlation)
- **C**: same as B + one contact sheet of the trained policy per candidate

Tasks: `CartpoleSwingup`, `CheetahRun` (both have a reference human reward), and `cheetah_stand`, a new task with no existing reward (rear up on the back leg and hold the pose).

Work in progress, results not in yet.

## Setup

```bash
uv venv -p 3.12 .venv && source .venv/bin/activate
uv pip install -e ".[dev]"
pytest
```

- LLM: `gemma4:31b` through Ollama (free cloud tier, accepts images). Locally the Ollama daemon must be signed in; elsewhere set `OLLAMA_HOST=https://ollama.com` and `OLLAMA_API_KEY`.
- Training: Kaggle GPU (2x T4) through the Kaggle CLI, one worker per GPU. The LLM side stays local.

## Usage

```bash
reward-lab smoke                                   # whole loop on CPU, tiny budgets
reward-lab loop   --exp configs/experiment.yaml --backend kaggle
reward-lab finals --exp configs/experiment.yaml --backend kaggle
reward-lab analyze --exp configs/experiment.yaml
```

Adding a task = a YAML file in `configs/tasks/` + a fitness function in `src/reward_lab/tasks.py`.

## Layout

```
configs/      tasks and experiment settings
prompts/      system prompt, task prompt, feedback prompt, fix prompt
src/reward_lab/
  context.py     named quantities the reward can read (the "API" given to the LLM)
  reward_env.py  wrapper that swaps the env reward and logs components + fitness
  sandbox.py     code extraction, restricted exec, jit/shape/NaN checks
  llm.py         Ollama client with on-disk cache
  train.py       Brax PPO with Playground's config, final eval rollout
  feedback.py    text summary and contact sheets
  loop.py        the loop itself, resumable from results/
  remote.py      runs a batch of trainings on Kaggle
  analysis.py    tables and figures
tests/
```
