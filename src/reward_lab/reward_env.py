"""Wrapper that swaps the env reward for an arbitrary compute_reward(ctx).

Each reward component and the per-step fitness are written to state.metrics.
Brax's evaluator sums metrics over an episode, so we get per-component and
fitness curves during training for free (eval/episode_comp/<name>, eval/episode_fitness).
"""

import jax.numpy as jp
from mujoco_playground import registry
from mujoco_playground._src.wrapper import Wrapper

from .context import make_builder


class RewardEnv(Wrapper):
    def __init__(self, env, task, reward_fn, comp_names):
        super().__init__(env)
        self._build = make_builder(task.family, env.mj_model)
        self._fitness = task.fitness
        self._reward_fn = reward_fn
        self._comp_names = list(comp_names)

    def reset(self, rng):
        state = self.env.reset(rng)
        # Keep the base env metrics: some envs (cartpole) write into them
        # in-place during step, so the pytree structure must not change.
        metrics = dict(state.metrics)
        metrics.update({f"comp/{k}": jp.zeros(()) for k in self._comp_names})
        metrics["fitness"] = jp.zeros(())
        return state.replace(reward=jp.zeros(()), metrics=metrics)

    def step(self, state, action):
        state = self.env.step(state, action)
        ctx = self._build(state.data, action)
        reward, comps = self._reward_fn(ctx)
        # A single NaN would poison the whole PPO batch; validation already
        # rejects rewards that produce NaN on sample states, this is the backstop.
        reward = jp.where(jp.isfinite(reward), reward, 0.0)
        metrics = dict(state.metrics)
        for k in self._comp_names:
            metrics[f"comp/{k}"] = jp.where(jp.isfinite(comps[k]), comps[k], 0.0)
        metrics["fitness"] = self._fitness(ctx)
        return state.replace(reward=reward, metrics=metrics)


def make_env(task, reward_fn, comp_names, impl="jax"):
    base = registry.load(task.env, config_overrides={"impl": impl})
    return RewardEnv(base, task, reward_fn, comp_names)
