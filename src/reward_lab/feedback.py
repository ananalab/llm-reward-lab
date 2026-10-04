"""What the LLM gets back between two iterations.

Text (conditions B and C): per candidate, the task score, how each reward
component evolved during training, and the correlation between the reward
and the task score across checkpoints (a cheap hacking indicator).

Images (condition C only): a contact sheet of the final policy, rendered from
the saved joint trajectory with the CPU MuJoCo renderer, so the GPU worker
never needs an OpenGL context.
"""

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from mujoco_playground import registry

CAMERA = {"cartpole": "fixed", "cheetah": "side"}
_models = {}


def _model(env_name):
    if env_name not in _models:
        _models[env_name] = registry.load(env_name, config_overrides={"impl": "jax"}).mj_model
    return _models[env_name]


def render_frames(task, qpos, idx, size=(320, 240)):
    m = _model(task.env)
    d = mujoco.MjData(m)
    frames = []
    with mujoco.Renderer(m, height=size[1], width=size[0]) as r:
        for t in idx:
            d.qpos[:] = qpos[t]
            mujoco.mj_forward(m, d)
            r.update_scene(d, camera=CAMERA[task.family])
            frames.append(r.render().copy())
    return frames


def _font(size):
    for f in ("/System/Library/Fonts/Supplemental/Arial.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(f, size)
        except OSError:
            pass
    return ImageFont.load_default()


def contact_sheet(task, traj, title, n=8, cols=4, dt=0.01):
    """n frames evenly spaced over the episode, captioned with time and the
    running task score up to that frame."""
    qpos, fit = traj["qpos"], traj["fitness"]
    T = len(qpos)
    idx = np.linspace(T // (2 * n), T - 1, n).astype(int)
    frames = render_frames(task, qpos, idx)
    h, w = frames[0].shape[:2]
    rows = (n + cols - 1) // cols
    head, cap = 30, 22
    sheet = Image.new("RGB", (cols * w, head + rows * (h + cap)), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((8, 6), title, fill="black", font=_font(18))
    for i, (t, fr) in enumerate(zip(idx, frames)):
        x, y = (i % cols) * w, head + (i // cols) * (h + cap)
        sheet.paste(Image.fromarray(fr), (x, y))
        label = f"t={t * dt:.1f}s  score so far={fit[: t + 1].mean():.2f}"
        draw.text((x + 6, y + h + 3), label, fill="black", font=_font(14))
    return sheet


def save_gif(task, traj, path, every=4, size=(320, 240), dt=0.01):
    idx = np.arange(0, len(traj["qpos"]), every)
    frames = [Image.fromarray(f) for f in render_frames(task, traj["qpos"], idx, size)]
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=int(1000 * dt * every), loop=0)


def _fmt(x):
    return f"{x:.3g}"


def summarize(metrics, comps):
    """Eureka-style text block for one trained candidate."""
    if metrics.get("status") != "ok":
        return "Training crashed: " + metrics.get("error", "").strip().splitlines()[-1][:300]
    curve = metrics["curve"]
    lines = [f"task score (final policy): {_fmt(metrics['fitness'])}",
             f"total reward per step (final policy): {_fmt(metrics['reward'])}"]
    steps = ", ".join(f"{c['step'] / 1e6:.0f}M" for c in curve)
    lines.append(f"checkpoints: [{steps}] (training steps)")
    lines.append("task score at checkpoints: [" + ", ".join(_fmt(c["fitness"]) for c in curve) + "]")
    lines.append("total reward per step at checkpoints: [" + ", ".join(_fmt(c["reward"]) for c in curve) + "]")
    for k in comps:
        vals = [c[f"comp/{k}"] for c in curve]
        lines.append(f"component `{k}` (mean per step) at checkpoints: [" + ", ".join(_fmt(v) for v in vals) + "]")
    r = np.array([c["reward"] for c in curve])
    f = np.array([c["fitness"] for c in curve])
    if r.std() > 1e-8 and f.std() > 1e-8:
        lines.append(f"correlation between total reward and task score across checkpoints: {np.corrcoef(r, f)[0, 1]:.2f}")
    return "\n".join(lines)
