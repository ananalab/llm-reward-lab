"""RewardContext: the named quantities a reward function is allowed to read.

The LLM never sees MuJoCo internals (qpos indices, body ids, xmat layout).
It gets a flat dict of physically meaningful quantities, each documented with
its unit and shape. The same documentation is pasted into the prompt, so the
spec below is the single source of truth for both the code and the LLM.

Everything here runs inside jit/vmap, one environment at a time.
"""

import jax.numpy as jp
import mujoco
from mujoco import mjx

# name -> (unit, shape, description)
CARTPOLE_FIELDS = {
    "cart_position": ("m", "()", "Cart position on the rail. 0 is the centre; the rail ends at -1.8 and +1.8."),
    "cart_velocity": ("m/s", "()", "Cart velocity, positive to the right."),
    "pole_angle": ("rad", "()", "Pole angle from the upright position, wrapped to [-pi, pi]. 0 = pointing straight up, +-pi = hanging down."),
    "pole_angular_velocity": ("rad/s", "()", "Angular velocity of the pole."),
    "pole_cos": ("-", "()", "cos(pole_angle): 1 when upright, -1 when hanging down."),
    "pole_tip_height": ("m", "()", "Height of the pole tip relative to the cart (pole length is 1 m): +1 upright, -1 hanging."),
    "action": ("-", "(1,)", "Motor command applied this step, in [-1, 1] (force on the cart)."),
}

CHEETAH_FIELDS = {
    "torso_x": ("m", "()", "Horizontal position of the torso along the running direction (distance travelled since reset)."),
    "torso_height": ("m", "()", "Height of the torso centre above the ground. About 0.7 when standing normally on four legs."),
    "torso_pitch": ("rad", "()", "Pitch of the torso in (-pi, pi]. 0 = horizontal, +pi/2 = vertical with the head up, -pi/2 = vertical with the head down, +-pi = upside down."),
    "forward_velocity": ("m/s", "()", "Forward (x) velocity of the centre of mass of the whole body. Positive = running forward."),
    "vertical_velocity": ("m/s", "()", "Vertical velocity of the torso, positive upward."),
    "pitch_rate": ("rad/s", "()", "Angular velocity of the torso pitch (positive = nose rotating up)."),
    "head_height": ("m", "()", "Height of the head above the ground."),
    "back_foot_height": ("m", "()", "Height of the back foot above the ground (about 0.05 when touching the ground)."),
    "front_foot_height": ("m", "()", "Height of the front foot above the ground (about 0.05 when touching the ground)."),
    "joint_angles": ("rad", "(6,)", "Leg joint angles, order: back thigh, back shin, back foot, front thigh, front shin, front foot."),
    "joint_velocities": ("rad/s", "(6,)", "Leg joint angular velocities, same order as joint_angles."),
    "action": ("-", "(6,)", "Motor commands applied this step, each in [-1, 1], same order as joint_angles."),
}

FIELDS = {"cartpole": CARTPOLE_FIELDS, "cheetah": CHEETAH_FIELDS}


def make_builder(family: str, mj_model: mujoco.MjModel):
    """Returns build(data, action) -> dict for the given robot family."""
    if family == "cartpole":
        slider = mj_model.jnt_qposadr[mj_model.joint("slider").id]
        hinge = mj_model.jnt_qposadr[mj_model.joint("hinge_1").id]
        pole = mj_model.body("pole_1").id

        def build(data: mjx.Data, action):
            cos = data.xmat[pole, 2, 2]
            sin = data.xmat[pole, 0, 2]
            return {
                "cart_position": data.qpos[slider],
                "cart_velocity": data.qvel[slider],
                "pole_angle": jp.arctan2(sin, cos),
                "pole_angular_velocity": data.qvel[hinge],
                "pole_cos": cos,
                "pole_tip_height": cos,
                "action": action,
            }

        return build

    if family == "cheetah":
        torso = mj_model.body("torso").id
        head = mj_model.geom("head").id
        bfoot = mj_model.geom("bfoot").id
        ffoot = mj_model.geom("ffoot").id
        sensor = mj_model.sensor("torso_subtreelinvel")
        vel_adr = mj_model.sensor_adr[sensor.id]

        # Heights are geom centres; capsules have radius 0.046, hence the
        # "about 0.05 when touching the ground" in the docs.
        def build(data: mjx.Data, action):
            xaxis = data.xmat[torso, :, 0]
            return {
                "torso_x": data.qpos[0],
                "torso_height": data.xpos[torso, 2],
                "torso_pitch": jp.arctan2(xaxis[2], xaxis[0]),
                "forward_velocity": data.sensordata[vel_adr],
                "vertical_velocity": data.qvel[1],
                # rooty rotates about +y, which tips the nose down; flip the sign
                # so that positive means nose up, consistent with torso_pitch.
                "pitch_rate": -data.qvel[2],
                "head_height": data.geom_xpos[head, 2],
                "back_foot_height": data.geom_xpos[bfoot, 2],
                "front_foot_height": data.geom_xpos[ffoot, 2],
                "joint_angles": data.qpos[3:9],
                "joint_velocities": data.qvel[3:9],
                "action": action,
            }

        return build

    raise ValueError(family)


def describe(family: str) -> str:
    """Markdown table of the context fields, pasted into the prompt."""
    rows = ["| name | unit | shape | meaning |", "|---|---|---|---|"]
    for name, (unit, shape, doc) in FIELDS[family].items():
        rows.append(f"| `{name}` | {unit} | {shape} | {doc} |")
    return "\n".join(rows)
