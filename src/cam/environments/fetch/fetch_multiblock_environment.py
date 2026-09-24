"""Fetch pick-and-place scene with several blocks, registered as "FetchMultiBlock-v0".

    gym.make("FetchMultiBlock-v0", num_blocks=3)

The stock Fetch scene is copied at runtime with bodies object0..object{N-1}
added. Asset references point into the installed gymnasium-robotics asset tree,
so no assets are copied. Each reset places every block on the table, apart
from the others.

Observation["observation"] layout (length 25 + 9 * (num_blocks - 1)):
    0:25                      standard Fetch observation (block0 is "object0")
    then for i in 1..N-1:     block i position (3), position relative to gripper (3), rotation (3)
"""

import os
import tempfile

import gymnasium as gym
import gymnasium_robotics
import numpy as np
from gymnasium.utils.ezpickle import EzPickle
from gymnasium_robotics.envs.fetch import MujocoFetchEnv
from gymnasium_robotics.utils import rotations

ENVIRONMENT_ID = "FetchMultiBlock-v0"

BLOCK_HALF_SIZE = 0.025
MIN_BLOCK_SEPARATION = 0.10
MIN_GRIPPER_CLEARANCE = 0.10
BLOCK_COLORS = ["1 0 0 1", "0 1 0 1", "0 0.4 1 1", "1 1 0 1", "1 0 1 1"]
MAX_BLOCKS = len(BLOCK_COLORS)


def build_multiblock_xml(num_blocks: int) -> str:
    assets = os.path.join(os.path.dirname(gymnasium_robotics.__file__), "envs", "assets")
    fetch_dir = os.path.join(assets, "fetch")
    blocks = "".join(
        f"""
		<body name="object{i}" pos="{1.25 + 0.06 * i} 0.53 0.425">
			<joint name="object{i}:joint" type="free" damping="0.01"></joint>
			<geom size="{BLOCK_HALF_SIZE} {BLOCK_HALF_SIZE} {BLOCK_HALF_SIZE}" type="box" condim="3" name="object{i}" rgba="{BLOCK_COLORS[i]}" mass="2"></geom>
			<site name="object{i}" pos="0 0 0" size="0.02 0.02 0.02" rgba="{BLOCK_COLORS[i]}" type="sphere"></site>
		</body>"""
        for i in range(num_blocks)
    )
    return f"""<?xml version="1.0" encoding="utf-8"?>
<mujoco>
	<compiler angle="radian" coordinate="local" meshdir="{os.path.join(assets, "stls", "fetch")}" texturedir="{os.path.join(assets, "textures")}"></compiler>
	<option timestep="0.002">
		<flag warmstart="enable"></flag>
	</option>

	<include file="{os.path.join(fetch_dir, "shared.xml")}"></include>

	<worldbody>
		<geom name="floor0" pos="0.8 0.75 0" size="0.85 0.7 1" type="plane" condim="3" material="floor_mat"></geom>
		<body name="floor0" pos="0.8 0.75 0">
			<site name="target0" pos="0 0 0.5" size="0.02 0.02 0.02" rgba="1 0 0 1" type="sphere"></site>
		</body>

		<include file="{os.path.join(fetch_dir, "robot.xml")}"></include>

		<body pos="1.3 0.75 0.2" name="table0">
			<geom size="0.25 0.35 0.2" type="box" mass="2000" material="table_mat"></geom>
		</body>
{blocks}

		<light directional="true" ambient="0.2 0.2 0.2" diffuse="0.8 0.8 0.8" specular="0.3 0.3 0.3" castshadow="false" pos="0 0 4" dir="0 0 -1" name="light0"></light>
	</worldbody>

	<actuator>
		<position ctrllimited="true" ctrlrange="0 0.2" joint="robot0:l_gripper_finger_joint" kp="30000" name="robot0:l_gripper_finger_joint" user="1"></position>
		<position ctrllimited="true" ctrlrange="0 0.2" joint="robot0:r_gripper_finger_joint" kp="30000" name="robot0:r_gripper_finger_joint" user="1"></position>
	</actuator>
</mujoco>
"""


class FetchMultiBlockEnv(MujocoFetchEnv, EzPickle):
    def __init__(self, num_blocks: int = 2, reward_type: str = "sparse", **kwargs):
        if not 1 <= num_blocks <= MAX_BLOCKS:
            raise ValueError(f"num_blocks must be in [1, {MAX_BLOCKS}], got {num_blocks}")
        self.num_blocks = num_blocks

        with tempfile.NamedTemporaryFile("w", prefix="fetch_multiblock_", suffix=".xml", delete=False) as f:
            f.write(build_multiblock_xml(num_blocks))
        initial_qpos = {"robot0:slide0": 0.405, "robot0:slide1": 0.48, "robot0:slide2": 0.0}
        for i in range(num_blocks):
            initial_qpos[f"object{i}:joint"] = [1.25 + 0.06 * i, 0.53, 0.4, 1.0, 0.0, 0.0, 0.0]
        try:
            MujocoFetchEnv.__init__(
                self,
                model_path=f.name,
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
        finally:
            os.remove(f.name)
        EzPickle.__init__(self, num_blocks=num_blocks, reward_type=reward_type, **kwargs)

    def _reset_sim(self):
        self._mujoco.mj_resetData(self.model, self.data)
        self.data.time = self.initial_time
        self.data.qpos[:] = np.copy(self.initial_qpos)
        self.data.qvel[:] = np.copy(self.initial_qvel)
        if self.model.na != 0:
            self.data.act[:] = None

        for i, xy in enumerate(self._sample_layout()):
            self._set_block_position(i, np.array([xy[0], xy[1], self.height_offset]))

        self._mujoco.mj_forward(self.model, self.data)
        return True

    def _sample_layout(self) -> list[np.ndarray]:
        """Block table positions near the gripper, pairwise apart; resampled until all fit."""
        while True:
            layout = self._try_sample_layout()
            if layout is not None:
                return layout

    def _try_sample_layout(self, attempts_per_block: int = 100) -> list[np.ndarray] | None:
        gripper_xy = self.initial_gripper_xpos[:2]
        placed: list[np.ndarray] = []
        for _ in range(self.num_blocks):
            for _ in range(attempts_per_block):
                xy = gripper_xy + self.np_random.uniform(-self.obj_range, self.obj_range, size=2)
                if np.linalg.norm(xy - gripper_xy) >= MIN_GRIPPER_CLEARANCE and all(
                    np.linalg.norm(xy - other) >= MIN_BLOCK_SEPARATION for other in placed
                ):
                    placed.append(xy)
                    break
            else:
                return None
        return placed

    def _set_block_position(self, index: int, position: np.ndarray) -> None:
        joint = f"object{index}:joint"
        qpos = self._utils.get_joint_qpos(self.model, self.data, joint)
        qpos[:3] = position
        qpos[3:] = [1.0, 0.0, 0.0, 0.0]
        self._utils.set_joint_qpos(self.model, self.data, joint, qpos)
        self._utils.set_joint_qvel(self.model, self.data, joint, np.zeros(6))

    def _get_obs(self):
        obs = super()._get_obs()
        gripper = obs["observation"][0:3]
        parts = [obs["observation"]]
        for i in range(1, self.num_blocks):
            position = self._utils.get_site_xpos(self.model, self.data, f"object{i}").copy()
            rotation = rotations.mat2euler(self._utils.get_site_xmat(self.model, self.data, f"object{i}"))
            parts.extend([position, position - gripper, rotation])
        obs["observation"] = np.concatenate(parts)
        return obs


if ENVIRONMENT_ID not in gym.registry:
    gym.register(
        id=ENVIRONMENT_ID,
        entry_point=f"{__name__}:FetchMultiBlockEnv",
        max_episode_steps=100,
    )
