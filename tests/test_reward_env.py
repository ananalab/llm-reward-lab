"""The human rewards rewritten with the RewardContext API must match the
original Playground rewards exactly, on real states."""

import jax
import jax.numpy as jp
import numpy as np
import pytest
from mujoco_playground import registry

from reward_lab import sandbox
from reward_lab.reward_env import make_env
from reward_lab.tasks import load_task


@pytest.mark.parametrize("name", ["cartpole_swingup", "cheetah_run"])
def test_human_reward_matches_original(name):
    task = load_task(name)
    batch = sandbox.make_ctx_batch(task, n=8, steps=10)
    comps = sandbox.validate(task.human_reward, batch)
    ours = make_env(task, sandbox.make_reward_fn(task.human_reward, comps), comps)
    orig = registry.load(task.env, config_overrides={"impl": "jax"})

    keys = jax.random.split(jax.random.PRNGKey(1), 16)
    s1 = jax.vmap(ours.reset)(keys)
    s2 = jax.vmap(orig.reset)(keys)
    step1, step2 = jax.jit(jax.vmap(ours.step)), jax.jit(jax.vmap(orig.step))
    rng = np.random.default_rng(0)
    for _ in range(50):
        a = jp.asarray(rng.uniform(-1, 1, (16, orig.action_size)), dtype=jp.float32)
        s1, s2 = step1(s1, a), step2(s2, a)
        np.testing.assert_allclose(s1.reward, s2.reward, rtol=1e-5, atol=1e-6)


def test_sandbox_rejects_python_if():
    task = load_task("cartpole_swingup")
    batch = sandbox.make_ctx_batch(task, n=4, steps=2)
    code = """
def compute_reward(ctx):
    if ctx["pole_cos"] > 0.9:
        r = 1.0
    else:
        r = 0.0
    return r, {"r": r}
"""
    with pytest.raises(sandbox.RewardError, match="jp.where"):
        sandbox.validate(code, batch)


def test_sandbox_rejects_imports_and_bad_shapes():
    task = load_task("cartpole_swingup")
    batch = sandbox.make_ctx_batch(task, n=4, steps=2)
    with pytest.raises(sandbox.RewardError, match="not allowed"):
        sandbox.validate("import os\ndef compute_reward(ctx):\n    return 0.0, {}\n", batch)
    with pytest.raises(sandbox.RewardError, match="scalar"):
        sandbox.validate("def compute_reward(ctx):\n    return ctx['action'], {'a': 0.0}\n", batch)
    with pytest.raises(sandbox.RewardError, match="NaN"):
        sandbox.validate("def compute_reward(ctx):\n    r = jp.log(ctx['pole_cos'] - 2)\n    return r, {'r': r}\n", batch)


def test_extract_code_takes_last_block():
    ans = "```python\ndef compute_reward(ctx):\n    return 0.0, {}\n```\ntext\n```python\ndef compute_reward(ctx):\n    return 1.0, {}\n```"
    assert "1.0" in sandbox.extract_code(ans)
