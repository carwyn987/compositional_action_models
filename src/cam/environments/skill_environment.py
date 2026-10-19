"""Skill environment: sets up episodes of skills for training.

SkillEnvironment turns an environment that reports symbolic facts into an RL
task over one or more skills. Each episode is one grounded action of one skill
(e.g. putdown(block2)):

    reset  choose the episode's skill (options={"skill": name}, otherwise the
           next skill in turn, or by competence), reset the scene, run that skill's setup chain
           (one of skill.setup_skills, e.g. pickup before putdown) to reach a state
           where it applies, then choose a random grounding whose
           preconditions hold
    step   the episode ends (terminated) once the grounding's effects have held for
           success_hold_steps consecutive steps (settled success: the skill must stop in
           its goal state, not pass through it), with the scene at rest on each of those steps
           (at_rest: e.g. the moved block no longer moving, so lifting on and on through a
           height band is not stopping in it) and every fact over frame_predicates
           that the action does not mention unchanged since the episode started;
           the reward comes from the skill's reward function (SparseReward,
           1.0 on success and 0.0 otherwise, unless another is given) plus the
           shared reward terms for every skill (e.g. motion penalties); the
           episode also ends, unsuccessfully and with failure_penalty, as soon as
           check_failure returns a reason (e.g. a block was dropped or knocked aside)

Each setup skill is executed like an episode of its own: a random applicable
grounding is chosen and its policy acts until that grounding's effects hold.
If a setup skill has no applicable grounding or does not succeed within
max_steps_per_setup_skill, the scene is reset and setup starts over, up to
max_setup_attempts times.

TODO: execute setup skills with learned policies instead of scripted ones, so
that training links actions into chains.

Requires the wrapped environment to provide info["facts"] (ground predicates
true in the state) and info["objects"] (object name -> type) on reset and
step, e.g. FetchPredicateEvaluationWrapper(FetchEnvStateAnnotationWrapper(...)).

Setup steps go through the wrapped environment, so a TimeLimit inside it counts
them. Put the episode step limit outside: TimeLimit(SkillEnvironment(...), n).
"""

import logging
from collections import deque
from typing import Callable

import gymnasium as gym

from cam.domain.symbolic_action_model import GroundedSymbolicActionModel, groundings
from cam.policies.policy import Policy
from cam.rewards.reward_function import RewardFunction, SparseReward
from cam.skills.registry import build_skill
from cam.skills.skill import Skill

logger = logging.getLogger(__name__)

# Competence sampling: a skill's recent success is over its last this many episodes in this environment ...
COMPETENCE_WINDOW = 20
# ... and every skill keeps at least this fraction of its even share of the episodes, however well it does
COMPETENCE_MIN_SHARE = 0.5


class SkillEnvironment(gym.Wrapper):
    """Episodes of skills: setup chain on reset, success/termination/reward on step.

    Adds to info:
        skill                   the episode's skill name                               (reset, step)
        grounded_action_model   the episode's grounded action, e.g. putdown(block2)   (reset, step)
        is_success              1.0 once its effects hold, else 0.0                   (reset, step)
        setup                   [(grounded setup action, steps taken), ...]           (reset)
        reward_components       named terms of the step's reward                     (step)
    """

    def __init__(
        self,
        env: gym.Env,
        skills: list[Skill],
        setup_policies: dict[str, Policy] | None = None,
        reward_functions: dict[str, RewardFunction] | None = None,
        max_steps_per_setup_skill: int = 100,
        max_setup_attempts: int = 10,
        success_hold_steps: int = 1,
        check_failure: Callable[[GroundedSymbolicActionModel, dict, dict], str | None] | None = None,
        frame_predicates: frozenset[str] = frozenset(),
        shared_reward: RewardFunction | None = None,
        failure_penalty: float = 0.0,
        competence_sampling: bool = False,
        at_rest: Callable[[GroundedSymbolicActionModel, dict, dict], bool] | None = None,
    ):
        super().__init__(env)
        self.success_hold_steps = success_hold_steps
        # Checks every step whether the episode has failed: (grounded action, info at the episode's start, info
        # now) -> the reason if it has, None if it has not. A reason makes the step a failure (episode ends).
        self.check_failure = check_failure
        # (grounded action, info at the previous step, info now) -> whether the scene is at rest; settled success
        # needs it on every step the effects hold (None: always)
        self.at_rest = at_rest
        self.previous_info: dict = {}
        self.shared_reward = shared_reward  # reward terms added for every skill
        self.failure_penalty = failure_penalty  # subtracted on the step a failure ends the episode
        self.start_info: dict = {}
        self.frame_predicates = frame_predicates  # facts the action must not change unless it says so
        self.start_facts: frozenset = frozenset()
        self.steps_effects_held = 0
        if not skills:
            raise ValueError("SkillEnvironment needs at least one skill")
        self.skills = {skill.name: skill for skill in skills}
        self.setup_chains = {  # skill -> its setup chains (one chosen at random per episode)
            skill.name: [[build_skill(name, skill.config) for name in chain]
                         for chain in skill.setup_skills or ((),)]
            for skill in skills
        }
        self.setup_policies = setup_policies or {}
        missing = sorted(
            {setup.name for chains in self.setup_chains.values() for chain in chains for setup in chain
             if setup.name not in self.setup_policies}
        )
        if missing:
            raise ValueError(f"no setup policy for {missing}")
        self.reward_functions = {name: SparseReward() for name in self.skills} | (reward_functions or {})
        self.max_steps_per_setup_skill = max_steps_per_setup_skill
        self.max_setup_attempts = max_setup_attempts
        self.episodes_started = 0
        # competence_sampling: skills with low recent success get more episodes (see _select_skill); without it,
        # the skills take turns
        self.competence_sampling = competence_sampling
        self.recent_success = {name: deque(maxlen=COMPETENCE_WINDOW) for name in self.skills}
        self.selection_credit = dict.fromkeys(self.skills, 0.0)
        self.episode_open = False  # an episode started and has not been recorded as a success or failure yet
        self.skill: Skill | None = None
        self.grounded_action_model: GroundedSymbolicActionModel | None = None

    def reset(self, *, seed=None, options=None):
        options = dict(options or {})
        if self.episode_open:  # the last episode ended without success or failure: cut off by the time limit
            self._record_outcome(success=False)
        self.skill = self._select_skill(options.pop("skill", None))
        self.episodes_started += 1
        for attempt in range(self.max_setup_attempts):
            obs, info = self.env.reset(seed=seed if attempt == 0 else None, options=options or None)
            if attempt == 0:
                self._check_scene_has_objects_for(self.skill, info["objects"])
            obs, info, setup = self._run_setup_chain(obs, info)
            if setup is None:
                continue
            grounded = self._choose_grounding(self.skill, info)
            if grounded is None:
                logger.info("setup failed: no applicable grounding of %s after setup", self.skill.name)
                continue
            logger.info("episode start: %s (preconditions hold)", grounded)
            if grounded.effects_hold(info["facts"]):
                logger.warning("episode start: %s effects already hold", grounded)
            self.grounded_action_model = grounded
            self.steps_effects_held = 0
            self.start_facts, self.start_info = info["facts"], info
            self.previous_info = info
            if self.shared_reward is not None:
                self.shared_reward.reset(grounded, info)
            self.reward_functions[self.skill.name].reset(grounded, info)
            self.episode_open = True
            return obs, {**self._annotate(info, success=False), "setup": setup}
        raise RuntimeError(f"could not set up {self.skill.name} in {self.max_setup_attempts} attempts")

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)
        # A failure takes precedence: a step where the failure condition holds never counts toward success,
        # even if the effects hold then (e.g. a dropped block that lands in place), and ends the episode.
        failure = (self.check_failure(self.grounded_action_model, self.start_info, info)
                   if self.check_failure else None)
        failed = failure is not None
        held = (
            self.grounded_action_model.effects_hold(info["facts"])
            and self.grounded_action_model.frame_holds(self.start_facts, info["facts"], self.frame_predicates)
            and not failed
            and (self.at_rest is None or self.at_rest(self.grounded_action_model, self.previous_info, info))
        )
        self.previous_info = info
        self.steps_effects_held = self.steps_effects_held + 1 if held else 0
        success = self.steps_effects_held >= self.success_hold_steps
        reward_function = self.reward_functions[self.skill.name]
        reward, components = reward_function(self.grounded_action_model, action, info, success)
        if self.shared_reward is not None:
            shared, shared_components = self.shared_reward(self.grounded_action_model, action, info, success)
            reward, components = reward + shared, {**components, **shared_components}
        if failed:
            logger.info("episode failed: %s (%s)", self.grounded_action_model, failure)
            reward, components = reward - self.failure_penalty, {**components, "failure": -self.failure_penalty}
        info = {**self._annotate(info, success), "reward_components": components, "failed": failed, "failure": failure}
        if success or failed:
            self._record_outcome(success)
        return obs, reward, terminated or success or failed, truncated, info

    def _record_outcome(self, success: bool) -> None:
        self.recent_success[self.skill.name].append(float(success))
        self.episode_open = False

    def competence_shares(self) -> dict[str, float]:
        """Each skill's share of the next episodes under competence sampling: COMPETENCE_MIN_SHARE of its even share,
        plus the rest in proportion to its recent failure rate (1 - success over its last COMPETENCE_WINDOW episodes;
        a skill without episodes yet counts as failing). Even shares when no skill fails."""
        floor = COMPETENCE_MIN_SHARE / len(self.skills)
        need = {name: 1.0 - (sum(outcomes) / len(outcomes) if outcomes else 0.0)
                for name, outcomes in self.recent_success.items()}
        total = sum(need.values())
        if total == 0:
            return dict.fromkeys(self.skills, 1.0 / len(self.skills))
        return {name: floor + (1.0 - floor * len(self.skills)) * need[name] / total for name in self.skills}

    def _select_skill(self, skill_name: str | None) -> Skill:
        """The named skill; with no name, the next skill in turn, or with competence sampling, the skill furthest
        behind its competence share (smooth weighted round-robin: each episode every skill gains its share as
        credit, and the skill with the most credit runs and pays one episode), so the shares hold without sampling
        noise."""
        if skill_name is None:
            if not self.competence_sampling:
                return list(self.skills.values())[self.episodes_started % len(self.skills)]
            for name, share in self.competence_shares().items():
                self.selection_credit[name] += share
            chosen = max(self.selection_credit, key=self.selection_credit.get)
            self.selection_credit[chosen] -= 1.0
            return self.skills[chosen]
        if skill_name not in self.skills:
            raise ValueError(f"unknown skill {skill_name!r}; this environment runs {sorted(self.skills)}")
        return self.skills[skill_name]

    def _check_scene_has_objects_for(self, skill: Skill, objects: dict[str, str]) -> None:
        """Raise ValueError if the skill or a setup skill has no grounding at all over the scene's objects
        (e.g. stack needs two distinct blocks), since then setup could never succeed."""
        for needed in [*(setup for chain in self.setup_chains[skill.name] for setup in chain), skill]:
            if not groundings(needed.symbolic_action_model, objects):
                parameters = ", ".join(f"{p.name} - {p.type}" for p in needed.symbolic_action_model.parameters)
                raise ValueError(
                    f"{needed.name}({parameters}) needs distinct objects of these types; the scene has {objects}"
                )

    def _run_setup_chain(self, obs, info):
        """Execute each setup skill of one of the episode skill's setup chains (chosen at random) in order;
        setup=None if any fails."""
        chains = self.setup_chains[self.skill.name]
        setup = []
        for setup_skill in chains[self.np_random.integers(len(chains))]:
            grounded = self._choose_grounding(setup_skill, info)
            if grounded is None:
                logger.info("setup failed: no applicable grounding of %s", setup_skill.name)
                return obs, info, None
            logger.info("setup start: %s (preconditions hold)", grounded)
            if grounded.effects_hold(info["facts"]):
                logger.warning("setup start: %s effects already hold", grounded)
            policy = self.setup_policies[setup_skill.name]
            policy.reset()
            steps = 0
            while not grounded.effects_hold(info["facts"]):
                if steps == self.max_steps_per_setup_skill:
                    logger.info("setup failed: %s did not succeed within %d steps", grounded, steps)
                    return obs, info, None
                obs, _, _, _, info = self.env.step(policy(obs, info, grounded))
                steps += 1
            logger.info("setup succeeded: %s in %d steps", grounded, steps)
            setup.append((grounded, steps))
        return obs, info, setup

    def _choose_grounding(self, skill: Skill, info: dict) -> GroundedSymbolicActionModel | None:
        candidates = skill.applicable_groundings(info["facts"], info["objects"])
        return candidates[self.np_random.integers(len(candidates))] if candidates else None

    def _annotate(self, info: dict, success: bool) -> dict:
        return {
            **info,
            "skill": self.skill.name,
            "grounded_action_model": self.grounded_action_model,
            "is_success": float(success),
        }
