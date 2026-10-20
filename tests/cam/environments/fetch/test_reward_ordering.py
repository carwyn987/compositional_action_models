"""Every skill's shaped reward ranks its scripted policy above doing nothing and above the degenerate behaviours a shaped
reward can end up paying for, in the simulator: by a margin in return, and none of them succeeds.

The degenerate behaviours, each a way a trained policy can settle without finishing the skill:
    idle            no action at all
    grip and wait   the scripted grasp, then no motion
    back off        the scripted grasp, then let go and move away, keeping the stage bonuses
    pump            the scripted grasp, then lift slowly and lower fast, over and over (pays if progress is clipped)
    overshoot       the raised variants' scripted lift carried 7 cm past the middle of the raised band
    hover           stack and place-beside: the block carried over its target and held there, never lowered
    hold            putdown: the block held where the setup left it
    behind and stop the pushes: the scripted approach behind the block, then no motion
A reward change that makes one of them pay as much as the skill fails here, before a training run finds it. Doing
nothing in putdown can succeed (a zero gripper command lets go, and one setup holds the block 1 cm up), so only its
return is compared.

Control: with pickup's lift term clipped per step (at a quarter of its target), the pump scores 33.3 against the
scripted 34.3, and this test fails.
"""

import numpy as np
import pytest

from cam.environments.fetch.fetch_scripted_policies import FetchScriptedPickupRaisedPolicy, servo
from cam.skills.registry import SKILL_REGISTRY
from scripts.run_skill import SCRIPTED_POLICIES

EPISODES = 4
MARGIN = 5.0  # the scripted mean return must beat each degenerate one by this much (success alone pays 15)
GRASPING = ("pickup", "unstack", "pickup-raised", "unstack-raised")


def environment(skill):
    from main import parse_args
    from cam.experiments.environment_setup import setup_environment
    from cam.skills.registry import build_skill

    config = parse_args(["--skills", skill, "--num-blocks", "2", "--reward", "shaped", "--seed", "0"])
    return setup_environment(config, [build_skill(skill, config)])


class Scripted:
    """The skill's scripted policy as act(obs, info), with optional changes once it reaches a phase."""

    def __init__(self, skill, policy=None):
        self.policy = policy or SCRIPTED_POLICIES[skill]()
        self.policy.reset()

    def scripted(self, obs, info):
        return np.asarray(self.policy(obs, info, info["grounded_action_model"]), dtype=np.float64)

    def __call__(self, obs, info):
        return self.scripted(obs, info)


class FrozenIn(Scripted):
    """No motion (the gripper as scripted) once the scripted policy is in one of the phases."""

    def __init__(self, skill, phases):
        super().__init__(skill)
        self.phases = phases

    def __call__(self, obs, info):
        action = self.scripted(obs, info)
        if getattr(self.policy, "phase", None) in self.phases:
            action[:3] = 0.0
        return action


class BackOff(Scripted):
    """Once the scripted grasp is done (lift phase): open and move 8 cm sideways and 6 cm up."""

    target = None

    def __call__(self, obs, info):
        action = self.scripted(obs, info)
        if self.policy.phase != "lift":
            return action
        gripper = info["environment_state"].gripper_position
        self.target = gripper + [0.0, 0.08, 0.06] if self.target is None else self.target
        return servo(gripper, self.target, 1.0)


class Pump(Scripted):
    """Once the scripted grasp is done: holding the block, lift it slowly to 6 cm, lower it fast, and again."""

    going_up = True

    def __call__(self, obs, info):
        action = self.scripted(obs, info)
        if self.policy.phase != "lift":
            return action
        state = info["environment_state"]
        start = state.block_positions[info["grounded_action_model"].arguments[0]]
        height = start[2] - 0.425
        self.going_up = height < 0.06 if self.going_up else height < 0.012
        return np.array([0.0, 0.0, 0.2 if self.going_up else -1.0, -1.0])


class Overshoot(FetchScriptedPickupRaisedPolicy):
    def lift_target(self):
        return super().lift_target() + [0.0, 0.0, 0.07]


def degenerates(skill):
    """name -> (factory of act(obs, info), whether it must not succeed)."""
    behaviours = {"idle": (lambda: (lambda obs, info: np.zeros(4)), skill != "putdown")}
    if skill in GRASPING:
        behaviours |= {
            "grip and wait": (lambda: FrozenIn(skill, {"lift"}), True),
            "back off": (lambda: BackOff(skill), True),
            "pump": (lambda: Pump(skill), False),  # held low and still for a moment, it may count as a pickup
        }
    if skill.endswith("-raised"):
        behaviours["overshoot"] = (lambda: Scripted(skill, Overshoot()), True)
    if skill in ("stack", "place-beside"):
        behaviours["hover"] = (lambda: FrozenIn(skill, {"lower"}), True)
    if skill == "putdown":
        behaviours["hold"] = (lambda: (lambda obs, info: np.array([0.0, 0.0, 0.0, -1.0])), True)
    if skill.startswith("push"):
        behaviours["behind and stop"] = (lambda: FrozenIn(skill, {"push"}), True)
    return behaviours


def play(env, skill, make_act):
    """Mean return and successes over EPISODES episodes on fixed scenes."""
    returns, successes = [], 0
    for episode in range(EPISODES):
        act = make_act()
        obs, info = env.reset(seed=300 + episode, options={"skill": skill})
        total, done, truncated = 0.0, False, False
        while not (done or truncated):
            obs, reward, done, truncated, info = env.step(act(obs, info))
            total += reward
        returns.append(total)
        successes += int(info["is_success"])
    return float(np.mean(returns)), successes


@pytest.mark.integration
@pytest.mark.parametrize("skill", sorted(SKILL_REGISTRY))
def test_the_scripted_policy_outscores_every_degenerate_behaviour(skill):
    env = environment(skill)
    try:
        scripted_return, scripted_successes = play(env, skill, lambda: Scripted(skill))
        results = {name: (*play(env, skill, make), must_fail)
                   for name, (make, must_fail) in degenerates(skill).items()}
    finally:
        env.close()
    assert scripted_successes >= 1
    for name, (mean_return, successes, must_fail) in results.items():
        assert scripted_return > mean_return + MARGIN, (name, scripted_return, mean_return)
        assert not (must_fail and successes), (name, successes)
