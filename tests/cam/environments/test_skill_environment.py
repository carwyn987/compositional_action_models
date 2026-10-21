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


class PutdownAfterPickup(PutdownSkill):
    """putdown set up by plain pickup, which this symbolic stand-in can run (it has no raised)."""

    setup_skills = (("pickup",),)


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
def test_success_needs_the_effects_to_hold_for_success_hold_steps_in_a_row():
    """Settled success: the effects must hold 3 consecutive steps; breaking them restarts the count."""
    env = SkillEnvironment(TwoBlockFactsEnv(), [PickupSkill(PDDL_CONFIG)], success_hold_steps=3)
    _, info = env.reset(seed=0)
    target = info["grounded_action_model"].arguments[0]
    outcomes = [env.step(action)[2] for action in [grasp(target), WAIT, RELEASE, grasp(target), WAIT, WAIT]]
    assert outcomes == [False, False, False, False, False, True]


@pytest.mark.unit
def test_a_failure_found_by_check_failure_ends_the_episode():
    def released(grounded, start_info, info):
        return "released" if Predicate("gripper-empty", ()) in info["facts"] else None

    env = SkillEnvironment(TwoBlockFactsEnv(), [PickupSkill(PDDL_CONFIG)], check_failure=released,
                           failure_penalty=2.0)
    env.reset(seed=0)
    _, reward, terminated, _, info = env.step(WAIT)
    assert terminated and info["failed"] and info["failure"] == "released" and info["is_success"] == 0.0
    assert reward == -2.0 and info["reward_components"]["failure"] == -2.0


@pytest.mark.unit
def test_a_failure_overrides_effects_that_hold():
    """If the failure condition holds on a step where the effects hold, the episode fails."""
    env = SkillEnvironment(TwoBlockFactsEnv(), [PickupSkill(PDDL_CONFIG)],
                           check_failure=lambda grounded, start_info, info: "always")
    _, info = env.reset(seed=0)
    _, _, terminated, _, info = env.step(grasp(info["grounded_action_model"].arguments[0]))
    assert terminated and info["failed"] and info["is_success"] == 0.0


@pytest.mark.unit
def test_each_episode_uses_one_of_the_skills_setup_chains():
    class PutdownEitherWay(PutdownSkill):
        setup_skills = (("pickup",), ("pickup", "putdown", "pickup"))

    env = SkillEnvironment(TwoBlockFactsEnv(), [PutdownEitherWay(PDDL_CONFIG)],
                           {"pickup": GraspTarget(), "putdown": Release()})
    chains = {len(env.reset(seed=seed)[1]["setup"]) for seed in range(20)}
    assert chains == {1, 3}


class LiftBothEnv(TwoBlockFactsEnv):
    """TwoBlockFactsEnv plus action [9]: grasp block0 and lift block1 with it, so block1 leaves the table."""

    def step(self, action):
        self.both = action[0] == 9 or (getattr(self, "both", False) and action[0] >= 0)
        return super().step(np.array([1.0]) if action[0] == 9 else action)

    def _info(self):
        info = super()._info()
        if getattr(self, "both", False):
            info["facts"] = info["facts"] - {Predicate("on-table", ("block1",))}
        return info


@pytest.mark.unit
def test_effects_reached_by_breaking_the_frame_never_succeed():
    """pickup(block0) by lifting both blocks reaches pickup's effects but takes block1 off the table, which
    pickup does not mention: no success however long it is held. Released and picked up cleanly, it succeeds:
    the frame is compared with the episode's start, so a past violation does not stick."""
    def pickup_block0(frame):
        env = SkillEnvironment(LiftBothEnv(), [PickupSkill(PDDL_CONFIG)], success_hold_steps=2, frame_predicates=frame)
        _, info = env.reset(seed=0)
        while info["grounded_action_model"].arguments[0] != "block0":  # this test lifts block0
            _, info = env.reset()
        return env

    env = pickup_block0(frozenset())  # without a frame, lifting both blocks counts (the exploit is real)
    assert [env.step(action)[2] for action in [np.array([9.0]), WAIT]] == [False, True]
    env = pickup_block0(frozenset({"on-table", "clear", "holding", "gripper-empty"}))
    assert [env.step(action)[2] for action in [np.array([9.0]), WAIT, WAIT, WAIT]] == [False, False, False, False]
    assert [env.step(action)[2] for action in [RELEASE, grasp("block0"), WAIT]] == [False, False, True]


@pytest.mark.unit
def test_setup_chain_runs_declared_skills():
    """putdown set up by pickup: reset runs pickup, so putdown targets the held block."""
    env = SkillEnvironment(TwoBlockFactsEnv(), [PutdownAfterPickup(PDDL_CONFIG)], {"pickup": GraspTarget()})
    _, info = env.reset(seed=0)
    (setup_action, steps), = info["setup"]
    assert str(setup_action).startswith("pickup(") and steps == 1
    assert info["grounded_action_model"].arguments == setup_action.arguments


@pytest.mark.unit
def test_setup_chain_of_several_skills():
    """Setup skills run in order: pickup then putdown leaves a block on the table to pick up again."""

    class PickupAfterPickupAndPutdown(PickupSkill):
        setup_skills = (("pickup", "putdown"),)

    policies = {"pickup": GraspTarget(), "putdown": Release()}
    env = SkillEnvironment(TwoBlockFactsEnv(), [PickupAfterPickupAndPutdown(PDDL_CONFIG)], policies)
    _, info = env.reset(seed=0)
    assert [str(grounded).split("(")[0] for grounded, _ in info["setup"]] == ["pickup", "putdown"]
    assert str(info["grounded_action_model"]).startswith("pickup(")


@pytest.mark.unit
def test_missing_setup_policy_is_rejected():
    """Every declared setup skill needs a policy; the error is raised at construction."""
    with pytest.raises(ValueError):
        SkillEnvironment(TwoBlockFactsEnv(), [PutdownAfterPickup(PDDL_CONFIG)])


@pytest.mark.unit
def test_reset_raises_when_setup_never_succeeds():
    """A setup policy that never reaches its effects makes reset give up after max_setup_attempts."""
    env = SkillEnvironment(
        TwoBlockFactsEnv(), [PutdownAfterPickup(PDDL_CONFIG)], {"pickup": DoNothing()},
        max_steps_per_setup_skill=5, max_setup_attempts=3,
    )
    with pytest.raises(RuntimeError):
        env.reset(seed=0)


@pytest.mark.unit
def test_reset_chooses_only_feasible_groundings():
    """feasible rules groundings out (e.g. a push with a block in its path): reset only starts the others."""
    env = SkillEnvironment(TwoBlockFactsEnv(), [PickupSkill(PDDL_CONFIG)],
                           feasible=lambda grounded, info: grounded.arguments == ("block1",))
    for seed in range(10):
        _, info = env.reset(seed=seed)
        assert str(info["grounded_action_model"]) == "pickup(block1)"
    env = SkillEnvironment(TwoBlockFactsEnv(), [PickupSkill(PDDL_CONFIG)], max_setup_attempts=3,
                           feasible=lambda grounded, info: False)
    with pytest.raises(RuntimeError):
        env.reset(seed=0)


@pytest.mark.unit
def test_reset_option_selects_the_skill():
    """reset(options={"skill": name}) runs an episode of that skill, including its setup chain."""
    env = SkillEnvironment(
        TwoBlockFactsEnv(), [PickupSkill(PDDL_CONFIG), PutdownAfterPickup(PDDL_CONFIG)], {"pickup": GraspTarget()}
    )
    _, info = env.reset(seed=0, options={"skill": "putdown"})
    assert info["skill"] == "putdown" and len(info["setup"]) == 1
    _, info = env.reset(options={"skill": "pickup"})
    assert info["skill"] == "pickup" and info["setup"] == []


@pytest.mark.unit
def test_skills_take_turns_without_a_reset_option():
    """Without a skill option, episodes cycle through the skills in the order given."""
    env = SkillEnvironment(
        TwoBlockFactsEnv(), [PickupSkill(PDDL_CONFIG), PutdownAfterPickup(PDDL_CONFIG)], {"pickup": GraspTarget()}
    )
    assert [env.reset(seed=0)[1]["skill"] for _ in range(3)] == ["pickup", "putdown", "pickup"]


def episodes_per_skill(env, episodes: int) -> dict[str, int]:
    """Run episodes in which pickup always succeeds (grasp its block) and putdown always fails (wait until the
    next reset cuts it off, as the time limit would); count each skill's episodes."""
    counts = {"pickup": 0, "putdown": 0}
    for _ in range(episodes):
        _, info = env.reset(seed=0)
        counts[info["skill"]] += 1
        target = info["grounded_action_model"].arguments[0]
        env.step(grasp(target) if info["skill"] == "pickup" else WAIT)
    return counts


@pytest.mark.unit
@pytest.mark.parametrize("competence_sampling", [False, True])
def test_competence_sampling_gives_failing_skills_more_episodes(competence_sampling):
    """pickup always succeeds and putdown always fails. Control: taking turns gives each 20 of 40 episodes. With
    competence sampling putdown gets most of them, while pickup keeps at least its minimum (half its even share)."""
    from cam.environments.skill_environment import COMPETENCE_MIN_SHARE

    env = SkillEnvironment(TwoBlockFactsEnv(), [PickupSkill(PDDL_CONFIG), PutdownAfterPickup(PDDL_CONFIG)],
                           {"pickup": GraspTarget()}, competence_sampling=competence_sampling)
    counts = episodes_per_skill(env, 40)
    if not competence_sampling:
        assert counts == {"pickup": 20, "putdown": 20}
    else:
        assert counts["putdown"] >= 28 and counts["pickup"] >= COMPETENCE_MIN_SHARE / 2 * 40 - 1
        assert env.competence_shares() == {"pickup": pytest.approx(0.25), "putdown": pytest.approx(0.75)}


@pytest.mark.unit
def test_competence_shares_are_even_while_no_skill_fails():
    """Without failures (here: no episodes recorded as failed) every skill gets an even share."""
    env = SkillEnvironment(TwoBlockFactsEnv(), [PickupSkill(PDDL_CONFIG), PutdownAfterPickup(PDDL_CONFIG)],
                           {"pickup": GraspTarget()}, competence_sampling=True)
    for name in env.recent_success:
        env.recent_success[name].extend([1.0] * 5)
    assert env.competence_shares() == {"pickup": 0.5, "putdown": 0.5}


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
