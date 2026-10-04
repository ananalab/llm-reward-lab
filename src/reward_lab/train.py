"""PPO training of one reward candidate, plus a final evaluation rollout.

PPO hyper-parameters are Playground's tuned ones for the base env, untouched,
except num_timesteps (short budget for screening, full budget for finals) and
num_evals. Every candidate of a task gets exactly the same config and seed
schedule, so the only thing that changes is the reward.

Used as a worker: python -m reward_lab.train --jobs jobs.json --out DIR
"""

import argparse
import json
import pickle
import platform
import time
import traceback
from pathlib import Path

import jax
import jax.numpy as jp
import numpy as np
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from mujoco_playground import wrapper
from mujoco_playground.config import dm_control_suite_params

from . import sandbox
from .reward_env import make_env
from .tasks import load_task

SMOKE = dict(num_envs=16, batch_size=16, num_minibatches=4, unroll_length=10,
             num_updates_per_batch=2, num_eval_envs=8, episode_length=100)


def ppo_config(task, steps, num_evals, smoke=False):
    cfg = dm_control_suite_params.brax_ppo_config(task.env)
    cfg.num_timesteps = steps
    cfg.num_evals = num_evals
    if smoke:
        cfg.update(SMOKE)
    return cfg


def train_one(job, out_dir: Path, impl="jax", smoke=False):
    task = load_task(job["task"])
    code, comps = job["code"], job["comps"]
    reward_fn = sandbox.make_reward_fn(code, comps)
    env = make_env(task, reward_fn, comps, impl)
    eval_env = make_env(task, reward_fn, comps, impl)
    cfg = ppo_config(task, job["steps"], job["num_evals"], smoke)
    ep_len = cfg.episode_length

    curve = []
    t0 = time.time()

    def progress(step, m):
        n = float(m.get("eval/avg_episode_length", ep_len))
        row = {"step": int(step), "time": time.time() - t0,
               "reward": float(m["eval/episode_reward"]) / n,
               "fitness": float(m["eval/episode_fitness"]) / n}
        for k in comps:
            row[f"comp/{k}"] = float(m[f"eval/episode_comp/{k}"]) / n
        curve.append(row)
        print(job["key"], row["step"], f"r={row['reward']:.3f} f={row['fitness']:.3f}", flush=True)

    kw = dict(cfg)
    kw.pop("network_factory", None)
    make_inference_fn, params, _ = ppo.train(
        environment=env, eval_env=eval_env, wrap_env_fn=wrapper.wrap_for_brax_training,
        network_factory=ppo_networks.make_ppo_networks, progress_fn=progress,
        seed=job["seed"], **kw)
    train_time = time.time() - t0

    final = rollout(eval_env, make_inference_fn(params, deterministic=True), comps,
                    n_envs=8 if smoke else 32, length=ep_len, seed=job["seed"] + 1000)

    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "traj.npz", **final.pop("traj"))
    with open(out_dir / "params.pkl", "wb") as f:
        pickle.dump(jax.device_get(params), f)
    metrics = {
        "status": "ok", "key": job["key"], "task": task.name, "seed": job["seed"],
        "steps": job["steps"], "impl": impl, "smoke": smoke,
        "train_time_s": train_time, "jit_time_s": curve[1]["time"] if len(curve) > 1 else None,
        "fitness": final["fitness"], "fitness_std": final["fitness_std"],
        "reward": final["reward"], "final_comps": final["comps"],
        "curve": curve, "device": str(jax.devices()[0].device_kind),
        "host": platform.node(),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=1))
    return metrics


def rollout(env, policy, comps, n_envs, length, seed):
    """Deterministic policy, n_envs episodes. Keeps env 0's trajectory for rendering."""
    reset = jax.jit(jax.vmap(env.reset))
    step = jax.jit(jax.vmap(env.step))
    policy = jax.jit(jax.vmap(policy))
    state = reset(jax.random.split(jax.random.PRNGKey(seed), n_envs))
    keys = jax.random.split(jax.random.PRNGKey(seed + 1), n_envs)

    @jax.jit
    def body(carry, _):
        state, keys = carry
        keys, sub = jax.vmap(jax.random.split, out_axes=1)(keys)
        action, _ = policy(state.obs, sub)
        state = step(state, action)
        out = {"reward": state.reward, "fitness": state.metrics["fitness"],
               "qpos": state.data.qpos[0], "qvel": state.data.qvel[0]}
        out.update({f"comp/{k}": state.metrics[f"comp/{k}"] for k in comps})
        return (state, keys), out

    _, traj = jax.lax.scan(body, (state, keys), None, length=length)
    traj = jax.device_get(traj)
    per_env_fit = traj["fitness"].mean(0)
    return {
        "fitness": float(per_env_fit.mean()),
        "fitness_std": float(per_env_fit.std()),
        "reward": float(traj["reward"].mean()),
        "comps": {k: float(traj[f"comp/{k}"].mean()) for k in comps},
        "traj": {"qpos": traj["qpos"], "qvel": traj["qvel"],
                 "fitness": traj["fitness"][:, 0], "reward": traj["reward"][:, 0]},
    }


def run_jobs(jobs, out_root: Path, impl="jax", smoke=False):
    for job in jobs:
        out_dir = out_root / job["task"] / job["key"]
        if (out_dir / "metrics.json").exists():
            continue
        try:
            train_one(job, out_dir, impl, smoke)
        except Exception as e:  # noqa: BLE001 - one bad candidate must not kill the batch
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "metrics.json").write_text(json.dumps({
                "status": "error", "key": job["key"], "task": job["task"],
                "error": "".join(traceback.format_exception(e))[-3000:]}, indent=1))
            print("ERROR", job["key"], repr(e)[:300], flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--jobs", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--impl", default="jax")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--shard", default="0/1", help="i/n: run every n-th job starting at i")
    a = p.parse_args()
    i, n = map(int, a.shard.split("/"))
    jobs = json.loads(Path(a.jobs).read_text())[i::n]
    run_jobs(jobs, Path(a.out), a.impl, a.smoke)


if __name__ == "__main__":
    main()
