"""Multi-block Fetch environment for stack / unstack skills.

Gymnasium-Robotics ships only single-block Fetch tasks (one ``object0``).
Stacking needs several blocks in the scene, each with a stable identifier, and
a way to designate which block is moved (the *mover*) and onto which it is
placed (the *base*).

We build a custom MuJoCo model at runtime by copying the stock pick_and_place
model and adding ``object1..objectN-1`` bodies. The trick that keeps this
self-contained (no copying of meshes/textures, no edits to site-packages) is
that ``MujocoRobotEnv`` accepts an *absolute* ``model_path`` verbatim, and the
shared/robot includes reference assets by bare filename resolved through the
main model's ``<compiler meshdir/texturedir>``. So we point ``meshdir``,
``texturedir`` and every ``<include>`` at absolute paths inside the installed
gymnasium-robotics asset tree.
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
from gymnasium.envs.registration import register
from gymnasium.utils.ezpickle import EzPickle

import gymnasium_robotics
from gymnasium_robotics.envs.fetch import MujocoFetchEnv
from gymnasium_robotics.utils import rotations


# Geometry of the stock Fetch block (half the side length, metres).
OBJECT_HALF_SIZE = 0.025

# Distinct site colours so blocks are visually distinguishable when rendering.
_BLOCK_RGBA = [
    "1 0 0 1",      # red
    "0 1 0 1",      # green
    "0 0.4 1 1",    # blue
    "1 1 0 1",      # yellow
    "1 0 1 1",      # magenta
]


def _assets_dir() -> str:
    return os.path.join(os.path.dirname(gymnasium_robotics.__file__), "envs", "assets")


def build_multiblock_xml(num_blocks: int, assets_dir: str | None = None) -> str:
    """Return a pick_and_place-style MJCF string with ``num_blocks`` blocks.

    All asset references (meshdir/texturedir/includes) are absolute so the model
    can be loaded from any path.
    """
    assets = assets_dir or _assets_dir()
    fetch_dir = os.path.join(assets, "fetch")
    meshdir = os.path.join(assets, "stls", "fetch")
    texturedir = os.path.join(assets, "textures")
    shared = os.path.join(fetch_dir, "shared.xml")
    robot = os.path.join(fetch_dir, "robot.xml")

    block_bodies = []
    for i in range(num_blocks):
        rgba = _BLOCK_RGBA[i % len(_BLOCK_RGBA)]
        # Initial pos is irrelevant (overwritten on reset) but must be valid.
        x = 1.25 + 0.06 * i
        block_bodies.append(
            f'''
		<body name="object{i}" pos="{x} 0.53 0.425">
			<joint name="object{i}:joint" type="free" damping="0.01"></joint>
			<geom size="0.025 0.025 0.025" type="box" condim="3" name="object{i}" material="block_mat" mass="2"></geom>
			<site name="object{i}" pos="0 0 0" size="0.02 0.02 0.02" rgba="{rgba}" type="sphere"></site>
		</body>'''
        )

    return f'''<?xml version="1.0" encoding="utf-8"?>
<mujoco>
	<compiler angle="radian" coordinate="local" meshdir="{meshdir}" texturedir="{texturedir}"></compiler>
	<option timestep="0.002">
		<flag warmstart="enable"></flag>
	</option>

	<include file="{shared}"></include>

	<worldbody>
		<geom name="floor0" pos="0.8 0.75 0" size="0.85 0.7 1" type="plane" condim="3" material="floor_mat"></geom>
		<body name="floor0" pos="0.8 0.75 0">
			<site name="target0" pos="0 0 0.5" size="0.02 0.02 0.02" rgba="1 0 0 1" type="sphere"></site>
		</body>

		<include file="{robot}"></include>

		<body pos="1.3 0.75 0.2" name="table0">
			<geom size="0.25 0.35 0.2" type="box" mass="2000" material="table_mat"></geom>
		</body>
{"".join(block_bodies)}

		<light directional="true" ambient="0.2 0.2 0.2" diffuse="0.8 0.8 0.8" specular="0.3 0.3 0.3" castshadow="false" pos="0 0 4" dir="0 0 -1" name="light0"></light>
	</worldbody>

	<actuator>
		<position ctrllimited="true" ctrlrange="0 0.2" joint="robot0:l_gripper_finger_joint" kp="30000" name="robot0:l_gripper_finger_joint" user="1"></position>
		<position ctrllimited="true" ctrlrange="0 0.2" joint="robot0:r_gripper_finger_joint" kp="30000" name="robot0:r_gripper_finger_joint" user="1"></position>
	</actuator>
</mujoco>
'''


class MujocoFetchMultiBlockEnv(MujocoFetchEnv, EzPickle):
    """Fetch environment with ``num_blocks`` blocks for stack/unstack.

    Each reset randomly picks a ``mover_index`` (the block to manipulate) and a
    different ``base_index`` (for stacking onto / unstacking from). Both ids are
    appended to the observation as one-hot vectors so a policy can act on the
    correct, identified blocks.

    Observation layout (length ``25 + 9*(num_blocks-1) + 2*num_blocks``):
      [0:25]                      standard Fetch obs (gripper + object0 + ...)
      then for i in 1..N-1:       object_i pos(3), rel-to-grip(3), rot(3)
      then:                       mover one-hot(N), base one-hot(N)
    """

    def __init__(
        self,
        num_blocks: int = 2,
        start_stacked: bool = False,
        reward_type: str = "sparse",
        **kwargs,
    ):
        if num_blocks < 2:
            raise ValueError("MujocoFetchMultiBlockEnv requires num_blocks >= 2")

        # These must exist before super().__init__ because it calls _get_obs to
        # build the observation space.
        self.num_blocks = int(num_blocks)
        self.start_stacked = bool(start_stacked)
        self.mover_index = 0
        self.base_index = 1

        # Write the generated model to a temp file and load it by absolute path.
        xml = build_multiblock_xml(self.num_blocks)
        tmp = tempfile.NamedTemporaryFile(
            prefix="fetch_multiblock_", suffix=".xml", delete=False, mode="w"
        )
        tmp.write(xml)
        tmp.close()
        self._model_xml_path = tmp.name

        initial_qpos = {
            "robot0:slide0": 0.405,
            "robot0:slide1": 0.48,
            "robot0:slide2": 0.0,
        }
        for i in range(self.num_blocks):
            initial_qpos[f"object{i}:joint"] = [
                1.25 + 0.06 * i,
                0.53,
                0.4,
                1.0,
                0.0,
                0.0,
                0.0,
            ]

        MujocoFetchEnv.__init__(
            self,
            model_path=self._model_xml_path,
            has_object=True,
            block_gripper=False,
            n_substeps=20,
            gripper_extra_height=0.2,
            target_in_the_air=True,
            target_offset=0.0,
            obj_range=0.15,
            target_range=0.15,
            distance_threshold=0.05,
            initial_qpos=initial_qpos,
            reward_type=reward_type,
            **kwargs,
        )
        EzPickle.__init__(
            self,
            num_blocks=num_blocks,
            start_stacked=start_stacked,
            reward_type=reward_type,
            **kwargs,
        )

    # -- reset / spawn ----------------------------------------------------

    def _sample_block_xy(self, existing: list[np.ndarray]) -> np.ndarray:
        """Sample a table xy near the gripper, separated from existing blocks."""
        gripper_xy = self.initial_gripper_xpos[:2]
        for _ in range(100):
            xy = gripper_xy + self.np_random.uniform(-self.obj_range, self.obj_range, size=2)
            if np.linalg.norm(xy - gripper_xy) < 0.1:
                continue
            if all(np.linalg.norm(xy - other) >= 0.10 for other in existing):
                return xy
        # Fallback: accept whatever we have rather than loop forever.
        return xy

    def _set_block_xyz(self, index: int, xyz: np.ndarray) -> None:
        qpos = self._utils.get_joint_qpos(self.model, self.data, f"object{index}:joint")
        assert qpos.shape == (7,)
        qpos[:3] = xyz
        # Reset orientation to upright and zero any residual velocity.
        qpos[3:] = np.array([1.0, 0.0, 0.0, 0.0])
        self._utils.set_joint_qpos(self.model, self.data, f"object{index}:joint", qpos)
        self._utils.set_joint_qvel(
            self.model, self.data, f"object{index}:joint", np.zeros(6)
        )

    def _reset_sim(self):
        # Reset buffers (mirrors MujocoFetchEnv._reset_sim).
        self._mujoco.mj_resetData(self.model, self.data)
        self.data.time = self.initial_time
        self.data.qpos[:] = np.copy(self.initial_qpos)
        self.data.qvel[:] = np.copy(self.initial_qvel)
        if self.model.na != 0:
            self.data.act[:] = None

        # Pick which blocks to manipulate this episode.
        self.mover_index = int(self.np_random.integers(self.num_blocks))
        choices = [i for i in range(self.num_blocks) if i != self.mover_index]
        self.base_index = int(self.np_random.choice(choices))

        table_z = self.height_offset

        if self.start_stacked:
            # Base on the table; mover stacked on top; any others on the table.
            placed: dict[int, np.ndarray] = {}
            base_xy = self._sample_block_xy([])
            self._set_block_xyz(self.base_index, np.array([base_xy[0], base_xy[1], table_z]))
            placed[self.base_index] = base_xy
            self._set_block_xyz(
                self.mover_index,
                np.array([base_xy[0], base_xy[1], table_z + 2 * OBJECT_HALF_SIZE]),
            )
            placed[self.mover_index] = base_xy
            for i in range(self.num_blocks):
                if i in placed:
                    continue
                xy = self._sample_block_xy(list(placed.values()))
                self._set_block_xyz(i, np.array([xy[0], xy[1], table_z]))
                placed[i] = xy
        else:
            # All blocks resting on the table, separated.
            placed_xy: list[np.ndarray] = []
            for i in range(self.num_blocks):
                xy = self._sample_block_xy(placed_xy)
                self._set_block_xyz(i, np.array([xy[0], xy[1], table_z]))
                placed_xy.append(xy)

        self._mujoco.mj_forward(self.model, self.data)
        return True

    # -- observation ------------------------------------------------------

    def _get_obs(self):
        obs = super()._get_obs()
        base_obs = obs["observation"]
        grip_pos = base_obs[0:3]

        extra = [base_obs]
        for i in range(1, self.num_blocks):
            pos = self._utils.get_site_xpos(self.model, self.data, f"object{i}").copy()
            rel = pos - grip_pos
            rot = rotations.mat2euler(
                self._utils.get_site_xmat(self.model, self.data, f"object{i}")
            )
            extra.extend([pos, rel, rot])

        mover_onehot = np.zeros(self.num_blocks, dtype=np.float64)
        mover_onehot[self.mover_index] = 1.0
        base_onehot = np.zeros(self.num_blocks, dtype=np.float64)
        base_onehot[self.base_index] = 1.0
        extra.extend([mover_onehot, base_onehot])

        obs["observation"] = np.concatenate(extra).astype(np.float64)
        return obs


# Register the two skill-specific configurations. Guard against re-registration
# when this module is imported multiple times.
def _register() -> None:
    from gymnasium.envs.registration import registry

    specs = {
        "FetchStackEnv-v0": {"start_stacked": False},
        "FetchUnstackEnv-v0": {"start_stacked": True},
    }
    for env_id, extra in specs.items():
        if env_id in registry:
            continue
        register(
            id=env_id,
            entry_point="fetch_blockworld.multiblock_env:MujocoFetchMultiBlockEnv",
            kwargs={"num_blocks": 2, **extra},
            max_episode_steps=150,
        )


_register()
