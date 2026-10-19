"""The experiments: each research question's overnight runs, by name.

    ./scripts/start_overnight.sh --experiment rq3-repair     # run one (the usual way, inside tmux)
    .venv/bin/python scripts/overnight_runs.py --list        # every experiment: what it asks, and its settings

An experiment is a class. Its docstring says what it asks, how, and what to read in the results; its attributes are
the settings of the overnight runner (scripts/overnight_runs.py), each named as the setting in lower case. An
environment variable named as the setting overrides it for one invocation (e.g. SEEDS="0 1 2" to replicate); the
runner lists the overrides it used, and start_overnight.sh records them in run_info.txt.
"""

from cam.skills.registry import SKILL_REGISTRY, build_skill


class Experiment:
    """The training protocol, shared by every experiment unless it sets otherwise."""

    name: str
    run_groups: tuple[str, ...]  # the runner's groups, from: learnability, rq1, repair
    base_skills: tuple[str, ...]  # the training skills; held-out operators (repairs' TO) only appear in repair runs
    repairs: tuple[tuple[str, str], ...] = ()  # (FROM, TO): each rq1 run continued with FROM replaced by TO
    conditions: tuple[str, ...] = (  # operator embedding conditions (CONDITION_OPTIONS in the runner)
        "one-hot", "random", "multi-hot-trainable", "text",
        "compositional-tree", "compositional-slots", "compositional-geometric",
    )
    alias_conditions: tuple[str, ...] = ("one-hot", "random")  # conditions that also get the repair-hidden control
    repair_modes: tuple[str, ...] = ("isolated", "retain")  # main.py --repair-mode
    seeds: tuple[int, ...] = (0,)
    algorithm: str = "mtsac"  # main.py --algorithm: sac; mtsac, SAC with a separate entropy temperature per skill; ppo
    # main.py --skill-sampling: round-robin, or competence: skills with low recent success get more of the training
    # episodes (the same rule for every condition)
    skill_sampling: str = "competence"
    text_backend: str = "openai"  # the text condition and compositional name embeddings; mock: offline, no meaning
    learn_steps: int = 150_000  # a learnability run's steps
    # how often a learnability run evaluates: its steps to threshold are the from-scratch baseline (a repair's
    # transfer gain) and set rq1_steps="auto", so they need a finer resolution than eval_interval
    learn_eval_interval: int = 10_000
    # an rq1 run's steps, or "auto": rq1_budget_margin x the sum over base_skills of the steps each needed alone to
    # reach the threshold in its learnability run (one that never did counts as learn_steps); rq1 runs then wait
    # for the learnability runs
    rq1_steps: int | str = 500_000
    rq1_budget_margin: float = 1.5
    repair_steps: int = 150_000  # a repair run's most steps
    repair_eval_interval: int = 10_000  # how often a repair run evaluates every skill
    eval_interval: int = 50_000
    eval_episodes: int = 20
    save_interval: int = 50_000

    @classmethod
    def settings(cls) -> dict[str, object]:
        """Every setting, by its runner name (upper case)."""
        return {name.upper(): getattr(cls, name) for name in Experiment.__annotations__ if name != "name"}


# The multi-skill training set: every literal of every operator rq3-repair holds out appears in one of these, in a
# different context.
TRAINING_SKILLS = (
    "pickup", "putdown", "stack", "unstack", "pickup-raised",
    "push-left", "push-right", "push-front", "push-back", "push-right-front", "push-left-back", "push-right-back",
    "push-beside",
)


class Learnability(Experiment):
    """Can each skill be learned alone, and how fast?

    Every registered skill trained alone by its own policy (multi-hot embedding) for learn_steps steps. A skill that
    can't be learned alone makes every result that uses it meaningless, so this is the gate for the other
    experiments. Its steps to threshold are also the from-scratch baseline for a repair's transfer gain, the
    individually trained policies of RQ4, and what rq1_steps="auto" adds up.

    Read: review_runs.py sections 1-3 (steps to threshold and final success per skill); for a skill that doesn't
    learn, the failure reasons in its log (docs/next_steps.md §2).
    """

    name = "learnability"
    run_groups = ("learnability",)
    base_skills = tuple(SKILL_REGISTRY)


class RQ1(Experiment):
    """RQ1: does conditioning on the operator's structure help a shared multi-skill policy?

    One shared policy on the training skills per operator embedding condition and seed, each with the same budget:
    conditions that only tell the policy which skill to do (one-hot, random) against conditions that show the
    operator's literals (multi-hot, text, compositional). Compared per skill and condition on steps to threshold
    (in all steps and on the skill), AUC and final success. The learnability runs go first: a skill that can't be
    learned alone says nothing about the conditions.

    Read: review_runs.py sections 1-4. One seed shows anecdotes; a claim needs SEEDS="0 1 2" or more.
    """

    name = "rq1"
    run_groups = ("learnability", "rq1")
    base_skills = TRAINING_SKILLS


class RQ3Repair(Experiment):
    """RQ3: does structure let a policy carry out a repaired operator, zero-shot or with less training?

    Each repaired operator (TO) is held out of training, and every one of its literals (precondition or effect,
    sign, predicate and arguments) is trained in another skill: only the combination is new. So this measures how
    much a policy learns about a behaviour from one literal in one place of the operator. Each rq1 policy (one per
    condition and seed) is continued with FROM replaced by TO, training only TO (repair mode isolated):
      - zero-shot: TO's success at the swap, before any training on it;
      - steps to threshold on TO, against learning TO from scratch (its learnability run): the transfer gain;
      - forgetting: the other skills, evaluated and not trained, before and after.
    Controls:
      - repair-hidden (alias_conditions): TO gets exactly FROM's embedding. If it succeeds zero-shot, FROM's
        behaviour already satisfies TO, and the pair tests nothing (review_runs.py flags it: REPAIR INVALID);
      - one-hot gives TO an index no training used: its zero-shot success is what the state alone gives.
    A pair is only informative where the rq1 policy learned FROM and the skills that carry TO's literals: check
    their final success in the rq1 run first.

    Pairs:
      unstack -> unstack-raised     effect (not (raised ?o)) becomes (raised ?o): the change from pickup to
                                    pickup-raised, both trained
      stack -> place-beside         putdown's effects, and (beside ?o ?u) and (on-table ?u) from push-beside: several
                                    literals at once, so read it as a new-skill test
      push-left -> push-left-front  adds (at-front ?o) and (not (at-front ?o)), trained in push-front and in the
                                    other diagonals
    The runner refuses to start if the training skills leave a literal of a repaired operator untrained.

    Conditions: one of each kind (identity, literals, text, composition), and only isolated repairs, whose steps
    compare directly with learning from scratch; CONDITIONS and REPAIR_MODES add the others at the cost of a longer
    plan. Read: review_runs.py section 5.
    """

    name = "rq3-repair"
    run_groups = ("learnability", "rq1", "repair")
    base_skills = TRAINING_SKILLS
    repairs = (("unstack", "unstack-raised"), ("stack", "place-beside"), ("push-left", "push-left-front"))
    conditions = ("compositional-tree", "one-hot", "multi-hot-trainable", "text")
    repair_modes = ("isolated",)


EXPERIMENTS: dict[str, type[Experiment]] = {
    experiment.name: experiment for experiment in (Learnability, RQ1, RQ3Repair)
}


def literals(skill: str) -> set[str]:
    """The skill's operator literals, e.g. "precondition (on-table ?u)" or "effect (not (holding ?o))": where the
    literal is in the operator, its sign, its predicate and its arguments."""
    model = build_skill(skill, {"symbolic_action_model_format": "pddl"}).symbolic_action_model

    def text(predicate, negated: bool) -> str:
        atom = f"({' '.join((predicate.name, *predicate.args))})"
        return f"(not {atom})" if negated else atom

    return ({f"precondition {text(p.predicate, p.negated)}" for p in model.preconditions}
            | {f"effect {text(e.predicate, e.delete)}" for e in model.effects})


def untrained_literals(skill: str, training_skills) -> list[str]:
    """The skill's literals that no training skill has: what a policy reading the operator never had the chance to
    learn."""
    trained = set().union(*(literals(name) for name in training_skills))
    return sorted(literals(skill) - trained)
