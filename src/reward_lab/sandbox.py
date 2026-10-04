"""Turn an LLM answer into a validated, jit-able reward function.

This is not a security boundary (exec never is). It is a guard rail against
the usual mistakes, and the real isolation is that training runs in a
throwaway Kaggle/Colab VM. What we do:
  * keep only the last ```python block of the answer;
  * reject imports other than jax / jax.numpy, and any dunder access;
  * exec with a tiny namespace (jax, jp, tolerance, a few builtins);
  * jit + vmap the function on real states and check shapes and finiteness.
"""

import ast
import builtins
import re
import traceback

import jax
import jax.numpy as jp
import numpy as np
from mujoco_playground._src.reward import tolerance

SAFE_BUILTINS = {
    n: getattr(builtins, n)
    for n in ("abs", "min", "max", "float", "int", "bool", "range", "len", "dict",
              "tuple", "list", "zip", "enumerate", "sum", "isinstance", "pow",
              "round", "True", "False", "None", "ValueError")
}
ALLOWED_IMPORTS = {"jax", "jax.numpy", "math"}

CODE_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.S)


class RewardError(Exception):
    """Anything wrong with a candidate; the message is shown to the LLM."""


def extract_code(answer: str) -> str:
    blocks = CODE_RE.findall(answer)
    blocks = [b for b in blocks if "def compute_reward" in b]
    if not blocks:
        raise RewardError("No ```python code block defining compute_reward(ctx) was found in your answer.")
    return blocks[-1].strip() + "\n"


def _check_ast(code: str):
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        raise RewardError(f"SyntaxError: {e}") from None
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module]
            bad = [n for n in names if n not in ALLOWED_IMPORTS]
            if bad:
                raise RewardError(f"Import of {bad} is not allowed; only jax, jax.numpy (as jp) and math are available.")
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            raise RewardError("Access to dunder attributes is not allowed.")
        if isinstance(node, ast.Name) and node.id.startswith("__"):
            raise RewardError("Access to dunder names is not allowed.")
    if not any(isinstance(n, ast.FunctionDef) and n.name == "compute_reward" for n in tree.body):
        raise RewardError("The code must define a top-level function compute_reward(ctx).")


def load_reward(code: str):
    """exec the code in a restricted namespace and return compute_reward."""
    _check_ast(code)
    import math

    ns = {"jax": jax, "jp": jp, "jnp": jp, "math": math, "tolerance": tolerance,
          "__builtins__": {**SAFE_BUILTINS, "__import__": _restricted_import}}
    try:
        exec(compile(code, "<reward>", "exec"), ns)
    except Exception as e:  # noqa: BLE001 - anything goes back to the LLM
        raise RewardError(_short_tb(e, code)) from None
    return ns["compute_reward"]


def _restricted_import(name, *args, **kwargs):
    if name not in ALLOWED_IMPORTS:
        raise ImportError(f"import {name} is not allowed")
    return __import__(name, *args, **kwargs)


def _short_tb(e: BaseException, code: str, limit: int = 1500) -> str:
    """Error message for the LLM: the offending line(s) of its code + the exception.
    Memory addresses are masked so that the same error always gives the same
    prompt (the LLM cache key depends on it)."""
    src = code.splitlines()
    where = [f"line {fr.lineno}: {src[fr.lineno - 1].strip()}"
             for fr in traceback.extract_tb(e.__traceback__)
             if fr.filename == "<reward>" and 0 < fr.lineno <= len(src)]
    msg = f"{type(e).__name__}: {e}"
    msg = re.sub(r"0x[0-9a-fA-F]+", "0x?", msg)
    msg = re.sub(r"\n\s*\n", "\n", msg)[:limit]
    return "\n".join(where[-2:] + [msg])


def validate(code: str, ctx_batch: dict):
    """Run the reward on a batch of real states. Returns component names.

    ctx_batch: dict of arrays with a leading batch axis (see make_ctx_batch).
    """
    fn = load_reward(code)

    def single(ctx):
        out = fn(ctx)
        if not (isinstance(out, tuple) and len(out) == 2 and isinstance(out[1], dict)):
            raise RewardError("compute_reward must return a tuple (reward, components) where components is a dict.")
        r, comps = out
        r = jp.asarray(r, dtype=jp.float32)
        if r.shape != ():
            raise RewardError(f"The reward must be a scalar for one environment, got shape {r.shape}. "
                              "Reduce vectors with jp.sum / jp.mean.")
        clean = {}
        for k, v in comps.items():
            v = jp.asarray(v, dtype=jp.float32)
            if v.shape != ():
                raise RewardError(f"Component '{k}' must be a scalar, got shape {v.shape}.")
            clean[str(k)] = v
        return r, clean

    try:
        r, comps = jax.jit(jax.vmap(single))(ctx_batch)
        r = np.asarray(r)
    except RewardError:
        raise
    except Exception as e:  # noqa: BLE001
        msg = _short_tb(e, code)
        if "TracerBoolConversionError" in msg or "ConcretizationTypeError" in msg:
            msg += ("\nHint: Python `if`/`and`/`or`/`while` on arrays cannot be traced by jax.jit. "
                    "Use jp.where(cond, a, b), jp.logical_and, jp.minimum, etc.")
        raise RewardError(msg) from None
    if not comps:
        raise RewardError("The components dict is empty; return each reward term in it.")
    bad = [k for k, v in comps.items() if not np.all(np.isfinite(np.asarray(v)))]
    if not np.all(np.isfinite(r)) or bad:
        raise RewardError(f"The reward produced NaN or inf on valid states (components: {bad or 'total'}). "
                          "Guard divisions, logs and sqrt.")
    return sorted(comps)


def make_reward_fn(code: str, comp_names):
    """Same as validate's inner function but with a fixed component order."""
    fn = load_reward(code)

    def reward_fn(ctx):
        r, comps = fn(ctx)
        r = jp.asarray(r, dtype=jp.float32)
        comps = {k: jp.asarray(comps[k], dtype=jp.float32) for k in comp_names}
        return r, comps

    return reward_fn


def make_ctx_batch(task, n: int = 64, steps: int = 60, seed: int = 0):
    """Real states for validation: reset + random actions, on CPU with impl=jax."""
    from mujoco_playground import registry

    from .context import make_builder

    env = registry.load(task.env, config_overrides={"impl": "jax"})
    build = make_builder(task.family, env.mj_model)
    keys = jax.random.split(jax.random.PRNGKey(seed), n)
    state = jax.jit(jax.vmap(env.reset))(keys)
    step = jax.jit(jax.vmap(env.step))
    # Different envs stop at different times so the batch covers more states.
    stop = np.linspace(0, steps, n).astype(int)
    rng = np.random.default_rng(seed)
    ctxs = []
    for t in range(steps + 1):
        hit = np.where(stop == t)[0]
        if len(hit):
            a = jp.zeros((n, env.action_size))
            ctxs.append(jax.tree.map(lambda x: x[hit], jax.vmap(build)(state.data, a)))
        action = jp.asarray(rng.uniform(-1, 1, (n, env.action_size)), dtype=jp.float32)
        state = step(state, action)
    batch = jax.tree.map(lambda *xs: jp.concatenate(xs), *ctxs)
    batch["action"] = jp.asarray(rng.uniform(-1, 1, batch["action"].shape), dtype=jp.float32)
    return batch
