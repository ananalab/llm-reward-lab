# Task

{description}

## Reward context

`ctx` contains the following entries (one robot, one control step, jax arrays):

{context}

## Results of the previous reward functions

Each reward function below was used to train a policy with PPO ({steps} training steps). The "task score" measures how well the robot actually does the task; it is computed by us and is never seen by the policy. Higher is better. {score_doc}

{evidence}

{candidates}

## What to do

1. For each candidate above, describe in one or two sentences what the trained robot most likely does, and give a verdict:
   - `solves`: the robot does what the task asks;
   - `hacks`: the policy collects reward without doing the task, by exploiting a loophole in the reward;
   - `fails`: the robot neither does the task nor exploits the reward (for example it learned almost nothing).
   Use exactly this format, one line per candidate: `Candidate <n>: <description> VERDICT: <solves|hacks|fails>`
2. Write an improved `compute_reward(ctx)` that achieves a higher task score. You may start from the best candidate or from scratch. Explain your changes in a few sentences, then give the full code in exactly one ```python block.
