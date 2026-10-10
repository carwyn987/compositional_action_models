import gymnasium as gym
import numpy as np
import pytest

from cam.domain.symbols import Predicate
from cam.environments.skill_environment import SkillEnvironment
from cam.policies.policy import Policy
from cam.rewards.reward_function import RewardFunction
from cam.skills.blocksworld_skills import PickupSkill, PutdownSkill

PDDL_CONFIG = {"symbolic_action_model_format": "pddl"}
BLOCKS = ("block0", "block1")
WAIT, RELEASE = np.array([0.0]), np.array([-1.0])


def grasp(block: str) -> np.ndarray:
    return np.array([1.0 + BLOCKS.index(block)])


class TwoBlockFactsEnv(gym.Env):
    """Symbolic stand-in for Fetch with two blocks. Action [1 + i] grasps block i, [-1] releases."""

    observation_space = gym.spaces.Box(0, 1, (1,))
    action_space = gym.spaces.Box(-1, 3, (1,))

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.held = None
        return np.zeros(1, dtype=np.float32), self._info()

    def step(self, action):
        if action[0] >= 1 and self.held is None:
            self.held = BLOCKS[int(action[0]) - 1]
        elif action[0] < 0:
            self.held = None
        return np.zeros(1, dtype=np.float32), -1.0, False, False, self._info()

    def _info(self):
        facts = {Predicate("holding", (self.held,))} if self.held else {Predicate("gripper-empty", ())}
        facts |= {Predicate(p, (b,)) for p in ("on-table", "clear") for b in BLOCKS if b != self.held}
        return {"facts": frozenset(facts), "objects": {b: "block" for b in BLOCKS}}


class GraspTarget(Policy):
    def __call__(self, obs, info, grounded_action_model):
        return grasp(grounded_action_model.arguments[0])


class Release(Policy):
    def __call__(self, obs, info, grounded_action_model):
        return RELEASE


class DoNothing(Policy):
    def __call__(self, obs, info, grounded_action_model):
        return WAIT


@pytest.mark.unit
def test_reset_chooses_an_applicable_grounding():
    """A skill without setup targets one of its applicable groundings, reported in info."""
    env = SkillEnvironment(TwoBlockFactsEnv(), [PickupSkill(PDDL_CONFIG)])
    _, info = env.reset(seed=0)
    assert str(info["grounded_action_model"]) in {"pickup(block0)", "pickup(block1)"}
    assert info["setup"] == [] and info["is_success"] == 0.0


@pytest.mark.unit
def test_step_rewards_and_terminates_when_effects_hold():
    """Reward 0 until the grounding's effects hold; then reward 1, terminated, is_success."""
    env = SkillEnvironment(TwoBlockFactsEnv(), [PickupSkill(PDDL_CONFIG)])
    _, info = env.reset(seed=0)
    target = info["grounded_action_model"].arguments[0]

    _, reward, terminated, _, info = env.step(WAIT)
    assert (reward, terminated, info["is_success"]) == (0.0, False, 0.0)

    _, reward, terminated, _, info = env.step(grasp(target))
    assert (reward, terminated, info["is_success"]) == (1.0, True, 1.0)


@pytest.mark.unit
def test_setup_chain_runs_declared_skills():
    """putdown declares setup_skills=("pickup",): reset runs pickup, so putdown targets the held block."""
    env = SkillEnvironment(TwoBlockFactsEnv(), [PutdownSkill(PDDL_CONFIG)], {"pickup": GraspTarget()})
    _, info = env.reset(seed=0)
    (setup_action, steps), = info["setup"]
    assert str(setup_action).startswith("pickup(") and steps == 1
    assert info["grounded_action_model"].arguments == setup_action.arguments


@pytest.mark.unit
def test_setup_chain_of_several_skills():
    """Setup skills run in order: pickup then putdown leaves a block on the table to pick up again."""

    class PickupAfterPickupAndPutdown(PickupSkill):
        setup_skills = ("pickup", "putdown")

    policies = {"pickup": GraspTarget(), "putdown": Release()}
    env = SkillEnvironment(TwoBlockFactsEnv(), [PickupAfterPickupAndPutdown(PDDL_CONFIG)], policies)
    _, info = env.reset(seed=0)
    assert [str(grounded).split("(")[0] for grounded, _ in info["setup"]] == ["pickup", "putdown"]
    assert str(info["grounded_action_model"]).startswith("pickup(")


@pytest.mark.unit
def test_missing_setup_policy_is_rejected():
    """Every declared setup skill needs a policy; the error is raised at construction."""
    with pytest.raises(ValueError):
        SkillEnvironment(TwoBlockFactsEnv(), [PutdownSkill(PDDL_CONFIG)])


@pytest.mark.unit
def test_reset_raises_when_setup_never_succeeds():
    """A setup policy that never reaches its effects makes reset give up after max_setup_attempts."""
    env = SkillEnvironment(
        TwoBlockFactsEnv(), [PutdownSkill(PDDL_CONFIG)], {"pickup": DoNothing()},
        max_steps_per_setup_skill=5, max_setup_attempts=3,
    )
    with pytest.raises(RuntimeError):
        env.reset(seed=0)


@pytest.mark.unit
def test_reset_option_selects_the_skill():
    """reset(options={"skill": name}) runs an episode of that skill, including its setup chain."""
    env = SkillEnvironment(
        TwoBlockFactsEnv(), [PickupSkill(PDDL_CONFIG), PutdownSkill(PDDL_CONFIG)], {"pickup": GraspTarget()}
    )
    _, info = env.reset(seed=0, options={"skill": "putdown"})
    assert info["skill"] == "putdown" and len(info["setup"]) == 1
    _, info = env.reset(options={"skill": "pickup"})
    assert info["skill"] == "pickup" and info["setup"] == []


@pytest.mark.unit
def test_skills_take_turns_without_a_reset_option():
    """Without a skill option, episodes cycle through the skills in the order given."""
    env = SkillEnvironment(
        TwoBlockFactsEnv(), [PickupSkill(PDDL_CONFIG), PutdownSkill(PDDL_CONFIG)], {"pickup": GraspTarget()}
    )
    assert [env.reset(seed=0)[1]["skill"] for _ in range(3)] == ["pickup", "putdown", "pickup"]


@pytest.mark.unit
def test_unknown_skill_option_is_rejected():
    """Asking for a skill the environment was not built with is an error."""
    env = SkillEnvironment(TwoBlockFactsEnv(), [PickupSkill(PDDL_CONFIG)])
    with pytest.raises(ValueError):
        env.reset(options={"skill": "putdown"})


class ConstantReward(RewardFunction):
    def __call__(self, grounded_action_model, action, info, success):
        return 0.5, {"constant": 0.5}


@pytest.mark.unit
def test_reward_function_supplies_reward_and_components():
    """A skill's reward function sets the step reward and info["reward_components"]; success still terminates."""
    env = SkillEnvironment(TwoBlockFactsEnv(), [PickupSkill(PDDL_CONFIG)], reward_functions={"pickup": ConstantReward()})
    _, info = env.reset(seed=0)
    _, reward, terminated, _, info = env.step(grasp(info["grounded_action_model"].arguments[0]))
    assert (reward, terminated, info["reward_components"]) == (0.5, True, {"constant": 0.5})
