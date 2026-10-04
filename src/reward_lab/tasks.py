"""Tasks: a Playground env, a natural-language description, and a fitness.

The fitness is what we actually care about and is never shown to the policy.
It is computed per control step from the same RewardContext as the reward,
then averaged over the episode, so every fitness here is "mean over 1000 steps
of something".
"""

from dataclasses import dataclass
from pathlib import Path

import jax.numpy as jp
import yaml

ROOT = Path(__file__).resolve().parents[2]
TASK_DIR = ROOT / "configs" / "tasks"


def upright_fraction(ctx):
    # Pole within ~18 degrees of vertical.
    return (ctx["pole_cos"] > 0.95).astype(jp.float32)


def forward_speed(ctx):
    # Mean forward speed in m/s (distance in 10 s = 10 x this).
    return ctx["forward_velocity"]


def rear_up_fraction(ctx):
    # Torso between 60 and 110 degrees nose-up, and high enough that the
    # cheetah is standing on its back leg rather than sitting on its rear end.
    pitch = ctx["torso_pitch"]
    ok = (pitch > jp.pi / 3) & (pitch < jp.deg2rad(110.0)) & (ctx["torso_height"] > 0.7)
    return ok.astype(jp.float32)


FITNESS = {f.__name__: f for f in (upright_fraction, forward_speed, rear_up_fraction)}

# Reference rewards written by the Playground / dm_control authors, rewritten
# with the RewardContext API. They go through the same sandbox as LLM code.
HUMAN_REWARDS = {
    "cartpole_swingup": '''
def compute_reward(ctx):
    upright = (ctx["pole_cos"] + 1) / 2
    centered = (1 + tolerance(ctx["cart_position"], margin=2)) / 2
    small_control = tolerance(ctx["action"][0], margin=1, value_at_margin=0, sigmoid="quadratic")
    small_control = (4 + small_control) / 5
    small_velocity = (1 + tolerance(ctx["pole_angular_velocity"], margin=5)) / 2
    reward = upright * centered * small_control * small_velocity
    return reward, {"upright": upright, "centered": centered,
                    "small_control": small_control, "small_velocity": small_velocity}
''',
    "cheetah_run": '''
def compute_reward(ctx):
    run = tolerance(ctx["forward_velocity"], bounds=(10.0, float("inf")), margin=10.0,
                    value_at_margin=0, sigmoid="linear")
    return run, {"run": run}
''',
}


@dataclass
class Task:
    name: str
    env: str
    family: str
    description: str
    score_doc: str
    fitness_name: str
    success: float
    short_steps: int
    full_steps: int

    @property
    def fitness(self):
        return FITNESS[self.fitness_name]

    @property
    def human_reward(self):
        return HUMAN_REWARDS.get(self.name)


def load_task(name: str) -> Task:
    cfg = yaml.safe_load((TASK_DIR / f"{name}.yaml").read_text())
    return Task(
        name=name,
        env=cfg["env"],
        family=cfg["family"],
        description=" ".join(cfg["description"].split()),
        score_doc=cfg["score_doc"],
        fitness_name=cfg["fitness"],
        success=float(cfg["success"]),
        short_steps=int(cfg["short_steps"]),
        full_steps=int(cfg["full_steps"]),
    )
