"""Skill environment: sets up episodes of skills for training.

SkillEnvironment turns an environment that reports symbolic facts into an RL
task over one or more skills. Each episode is one grounded action of one skill
(e.g. putdown(block2)):

    reset  choose the episode's skill (options={"skill": name}, otherwise the
           next skill in turn), reset the scene, run that skill's setup chain
           (skill.setup_skills, e.g. pickup before putdown) to reach a state
           where it applies, then choose a random grounding whose
           preconditions hold
    step   the episode ends (terminated) once the grounding's effects hold;
           reward is 1.0 on that step, else 0.0

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

import gymnasium as gym

from cam.domain.symbolic_action_model import GroundedSymbolicActionModel
from cam.policies.policy import Policy
from cam.skills.registry import build_skill
from cam.skills.skill import Skill

logger = logging.getLogger(__name__)


class SkillEnvironment(gym.Wrapper):
    """Episodes of skills: setup chain on reset, success/termination/reward on step.

    Adds to info:
        skill                   the episode's skill name                               (reset, step)
        grounded_action_model   the episode's grounded action, e.g. putdown(block2)   (reset, step)
        is_success              1.0 once its effects hold, else 0.0                   (reset, step)
        setup                   [(grounded setup action, steps taken), ...]           (reset)
    """

    def __init__(
        self,
        env: gym.Env,
        skills: list[Skill],
        setup_policies: dict[str, Policy] | None = None,
        max_steps_per_setup_skill: int = 100,
        max_setup_attempts: int = 10,
    ):
        super().__init__(env)
        if not skills:
            raise ValueError("SkillEnvironment needs at least one skill")
        self.skills = {skill.name: skill for skill in skills}
        self.setup_chains = {
            skill.name: [build_skill(name, skill.config) for name in skill.setup_skills] for skill in skills
        }
        self.setup_policies = setup_policies or {}
        missing = sorted(
            {name for skill in skills for name in skill.setup_skills if name not in self.setup_policies}
        )
        if missing:
            raise ValueError(f"no setup policy for {missing}")
        self.max_steps_per_setup_skill = max_steps_per_setup_skill
        self.max_setup_attempts = max_setup_attempts
        self.episodes_started = 0
        self.skill: Skill | None = None
        self.grounded_action_model: GroundedSymbolicActionModel | None = None

    def reset(self, *, seed=None, options=None):
        options = dict(options or {})
        self.skill = self._select_skill(options.pop("skill", None))
        self.episodes_started += 1
        for attempt in range(self.max_setup_attempts):
            obs, info = self.env.reset(seed=seed if attempt == 0 else None, options=options or None)
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
            return obs, {**self._annotate(info, success=False), "setup": setup}
        raise RuntimeError(f"could not set up {self.skill.name} in {self.max_setup_attempts} attempts")

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)
        success = self.grounded_action_model.effects_hold(info["facts"])
        reward = 1.0 if success else 0.0
        return obs, reward, terminated or success, truncated, self._annotate(info, success)

    def _select_skill(self, skill_name: str | None) -> Skill:
        """The named skill, or the next skill in turn when no name is given."""
        if skill_name is None:
            return list(self.skills.values())[self.episodes_started % len(self.skills)]
        if skill_name not in self.skills:
            raise ValueError(f"unknown skill {skill_name!r}; this environment runs {sorted(self.skills)}")
        return self.skills[skill_name]

    def _run_setup_chain(self, obs, info):
        """Execute each setup skill of the episode's skill in order; setup=None if any fails."""
        setup = []
        for setup_skill in self.setup_chains[self.skill.name]:
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
