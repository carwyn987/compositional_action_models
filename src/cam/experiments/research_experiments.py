"""The experiments: each research question's overnight runs, by name.

    ./scripts/start_overnight.sh --experiment rq3-repair-minimal   # run one (the usual way, inside tmux)
    .venv/bin/python scripts/overnight_runs.py --list        # every experiment: what it asks, and its settings

An experiment is a class. Its docstring says what it asks, how, and which metrics answer it; its attributes are
the settings of the overnight runner (scripts/overnight_runs.py), each named as the setting in lower case. An
environment variable named as the setting overrides it for one invocation (e.g. SEEDS="0 1 2" to replicate); the
runner lists the overrides it used, and start_overnight.sh records them in run_info.txt.
"""

from cam.skills.registry import SKILL_REGISTRY, build_skill


class Experiment:
    """The training protocol, shared by every experiment unless it sets otherwise.

    The settings of every run group are here, not only in the experiments that use them: an experiment can run
    several groups (rq3-repair-minimal runs learnability, rq1 and repair runs), and the runner reads each group's settings
    from whichever experiment it runs. Settings of a group an experiment does not run are unused.

    Every run records the same metrics (metrics.json, summarized by review_runs.py), since computing them costs
    little next to training: each skill's success rate at every evaluation (deterministic and stochastic), its
    zero-shot success (its first evaluation), the steps and episodes until it reaches the success threshold (in all
    training steps, and in the skill's own), AUC (its mean success over the run), its final success, and, after a
    skill is added or repaired, how much the other skills forgot. Each experiment's docstring names the metrics that
    answer its question.
    """

    name: str
    run_groups: tuple[str, ...]  # the runner's groups, from: learnability, rq1, repair
    base_skills: tuple[str, ...]  # the training skills; held-out operators (repairs' TO) only appear in repair runs
    seeds: tuple[int, ...] = (0,)
    algorithm: str = "mtsac"  # main.py --algorithm: sac; mtsac, SAC with a separate entropy temperature per skill; ppo
    # main.py --skill-sampling: round-robin, or competence: skills with low recent success get more of the training
    # episodes (the same rule for every operator embedding)
    skill_sampling: str = "competence"
    text_backend: str = "openai"  # the text embedding and compositional name embeddings; mock: offline, no meaning
    eval_episodes: int = 20  # episodes per skill in each evaluation, deterministic and stochastic
    save_interval: int = 50_000

    # Learnability runs: each skill alone.
    # A learnability run's steps. Alone, the straight pushes need about 160k steps to reach the threshold and the
    # diagonals 240-290k, and stack, push-beside and push-left-back were still improving at 300k.
    learn_steps: int = 400_000
    # The operator embedding a learnability run's policy sees (one of the runner's OPERATOR_EMBEDDING_OPTIONS). One
    # skill alone always sees the same embedding, so the choice should not matter; multi-hot is fixed and cheap.
    learnability_embedding: str = "multi-hot"
    # How often a learnability run evaluates: its steps to threshold are the from-scratch baseline (a repair's transfer
    # gain) and set rq1_steps="auto", so they need a finer resolution than train_eval_interval.
    learn_eval_interval: int = 10_000
    train_eval_interval: int = 50_000  # how often an rq1 run evaluates every skill (learnability: learn_eval_interval)

    # Rq1 runs: one shared policy on base_skills per operator embedding and seed (also the policies repairs continue).
    # The operator embeddings to compare, each the input that tells the shared policy which skill to perform (the
    # runner's OPERATOR_EMBEDDING_OPTIONS): identifiers (one-hot, random) against the operator's literals (multi-hot,
    # text, compositional).
    operator_embeddings: tuple[str, ...] = (
        "one-hot", "random", "multi-hot-trainable", "text",
        "compositional-tree", "compositional-slots", "compositional-geometric",
    )
    # Whether the operator embeddings above that carry the operator's name (text, in its text; compositional, as a
    # component) see it, each value a variant of them: exclude (labelled <condition>-no-name; main.py
    # --operator-text-name none, --compositional-name none) or include (labelled <condition>). Held-out operators
    # share name words with training skills (unstack-raised and pickup-raised: "raised"), so with the name a zero-shot
    # success could come from a word rather than the operator's structure. "exclude include" runs both: the ablation.
    operator_names: tuple[str, ...] = ("exclude",)
    # An rq1 run's steps, or "auto": set from the learnability runs, rq1_budget_margin x the sum over base_skills of the
    # steps each needed alone to reach the threshold (one that never did counts as learn_steps). rq1 runs then wait for
    # the learnability runs.
    rq1_steps: int | str = 500_000
    # rq1_steps="auto" only: how much more than the sum of its skills' steps alone a shared policy gets. It splits its
    # steps over the skills and they can interfere, so the bare sum would shortchange it; 1.5 is 50% headroom.
    rq1_budget_margin: float = 1.5

    # Repair runs: each rq1 run continued with FROM replaced by TO.
    repairs: tuple[tuple[str, str], ...] = ()  # (FROM, TO) pairs
    # How a repair run trains (main.py --repair-mode): isolated trains only TO, and evaluates the others without
    # training them, so its steps to threshold compare directly with learning TO from scratch; retain trains every
    # skill, until TO is at the threshold and the others are back at their level before: the cost of a repair
    # without forgetting.
    repair_modes: tuple[str, ...] = ("isolated", "retain")
    # The operator embeddings that also get the repair-hidden control: a second repair run in which TO is encoded
    # exactly as FROM (main.py --operator-identity-aliases), so the policy cannot see the repair. If it succeeds
    # zero-shot anyway, FROM's behaviour already satisfies TO and the pair tests nothing. Only identifier embeddings
    # (one-hot, random) can encode one operator as another.
    repair_hidden_embeddings: tuple[str, ...] = ("one-hot", "random")
    repair_steps: int = 150_000  # a repair run's most steps
    repair_eval_interval: int = 10_000  # how often a repair run evaluates every skill

    @classmethod
    def settings(cls) -> dict[str, object]:
        """Every setting, by its runner name (upper case)."""
        return {name.upper(): getattr(cls, name) for name in Experiment.__annotations__ if name != "name"}


# The multi-skill training set: every literal of every operator the repair experiments hold out appears in one of
# these, in a different context.
TRAINING_SKILLS = (
    "pickup", "putdown", "stack", "unstack", "pickup-raised",
    "push-left", "push-right", "push-front", "push-back", "push-right-front", "push-left-back", "push-right-back",
    "push-beside",
)


class Learnability(Experiment):
    """Can each skill be learned alone, and how fast?

    Every registered skill trained alone by its own policy (learnability_embedding) for learn_steps steps. A skill
    that can't be learned alone makes every result that uses it meaningless, so this is the gate for the other
    experiments. Its steps to threshold are also the from-scratch baseline for a repair's transfer gain, the
    individually trained policies of RQ4, and what rq1_steps="auto" adds up.

    The metrics that answer it: each skill's deterministic steps to threshold and final success. A learnable skill
    reaches the threshold (review_runs.py section 2 shows a number of steps, not "never") and its curve (section 3)
    ends near 1. If a skill stays low, watch its policy (watch_best_policies.py --folder <OUT>/learnability) to see
    where it stops, and cross-reference the failure reasons in its log (<OUT>/logs/learnability_<skill>.log,
    "episode failed"): mostly failures point at the constraints (thresholds, penalties), mostly time-outs at the
    reward or the step limit (docs/next_steps.md §2).
    """

    name = "learnability"
    run_groups = ("learnability",)
    base_skills = tuple(SKILL_REGISTRY)


class RQ1(Experiment):
    """RQ1: does conditioning on the operator's structure help a shared multi-skill policy?

    One shared policy on the training skills per operator embedding and seed, each with the same budget: embeddings
    that only tell the policy which skill to do (one-hot, random) against embeddings that show the operator's
    literals (multi-hot, text, compositional). The learnability runs go first: a skill that can't be learned alone
    says nothing about the embeddings.

    The metrics that answer it, per skill and embedding, across seeds (one seed is an anecdote; a claim needs
    SEEDS="0 1 2" or more): steps to threshold in the skill's own steps (comparable across skills and skill
    sampling), AUC and final success. Compare per skill, not averaged: easy skills dominate a mean. If the identifier
    embeddings match the structured ones on every skill, the operator's structure bought nothing on training skills,
    which the literature predicts (docs/related_work.md); differences are most likely on skills whose literals other
    training skills share. The overlap report the runner prints first (<OUT>/overlap.txt; the overlap experiment)
    gives each skill's shared fraction: compare the embeddings' gains on skills with high and with low sharing.
    Check first, in the learnability results, that every training skill learns alone.
    """

    name = "rq1"
    run_groups = ("learnability", "rq1")
    base_skills = TRAINING_SKILLS


class RQ3Repair(Experiment):
    """RQ3: does structure let a policy carry out a repaired operator, zero-shot or with less training?

    Each repaired operator (TO) is held out of training, and every one of its literals (precondition or effect,
    sign, predicate and arguments) is trained in another skill: only the combination is new. So this measures how
    much a policy learns about a behaviour from one literal in one place of the operator. Each rq1 policy (one per
    operator embedding and seed) is continued with FROM replaced by TO, training only TO (repair mode isolated).
    Controls:
      - repair-hidden (repair_hidden_embeddings): TO gets exactly FROM's embedding. If it succeeds zero-shot, FROM's
        behaviour already satisfies TO, and the pair tests nothing (review_runs.py flags it: REPAIR INVALID);
      - one-hot gives TO an index no training used: its zero-shot success is what the state alone gives;
      - the text and compositional embeddings leave out the operator's name (operator_names), which shares words
        with training skills (unstack-raised and pickup-raised: "raised") and could carry a repair by itself;
        OPERATOR_NAMES="exclude include" adds the variants with the name, the ablation.

    The metrics that answer it (review_runs.py section 5): TO's zero-shot success at the swap, repair-visible against
    repair-hidden (the difference is what the policy read from the operator); TO's steps to threshold against its
    learnability run (the transfer gain); and what the other skills forgot. A pair is only informative where the rq1
    policy learned FROM and the skills that carry TO's literals, so check their final success in the rq1 run first.
    Coverage in the overlap report (printed first) says how much of TO the training skills have, and the runner
    refuses to start if the training skills leave a literal of a repaired operator untrained. Operator embeddings:
    one of each kind (identity, literals, text, composition), and only isolated repairs, whose steps compare directly
    with learning from scratch; OPERATOR_EMBEDDINGS and REPAIR_MODES add the others at the cost of a longer plan.
    Two experiments: rq3-repair-minimal (few training skills, each repair's new literals in one other skill) and
    rq3-repair-maximal (every training skill, one repair).
    """

    run_groups = ("learnability", "rq1", "repair")
    operator_embeddings = ("compositional-tree", "one-hot", "multi-hot-trainable", "text")
    repair_modes = ("isolated",)


class RQ3RepairMinimal(RQ3Repair):
    """RQ3, minimal: each repair's training side is FROM and the few skills that carry TO's new literals.

    Seven training skills, so the shared policy trains quickly, and each new literal is seen in exactly one other
    context. Read it for a faster repair (fewer steps to threshold than learning TO from scratch) more than for
    zero-shot success, which one other context makes less likely. The controls and the metrics are RQ3Repair's
    (research_experiments.py).

    Pairs, each written with every training skill it relies on (relies_on): together they have every literal of
    the operator on the right, so the repaired operator is made only of literals the policy has seen.
      unstack, pickup, pickup-raised -> unstack-raised
          FROM is unstack; effect (not (raised ?o)) becomes (raised ?o), which pickup-raised adds, and pickup is
          its contrast (the same operator with (not (raised ?o)))
      putdown, push-beside -> place-beside
          FROM is putdown; adds parameter ?u, precondition (on-table ?u) and effect (beside ?o ?u), all in
          push-beside
      push-left, push-front -> push-left-front
          FROM is push-left; adds precondition (not (at-front ?o)) and effect (at-front ?o), both in push-front
    """

    name = "rq3-repair-minimal"
    base_skills = ("pickup", "pickup-raised", "unstack", "putdown", "push-beside", "push-left", "push-front")
    repairs = (("unstack", "unstack-raised"), ("putdown", "place-beside"), ("push-left", "push-left-front"))
    # each repaired operator (TO) -> the training skills it relies on, FROM first (documentation, checked by a test)
    relies_on = {
        "unstack-raised": ("unstack", "pickup", "pickup-raised"),
        "place-beside": ("putdown", "push-beside"),
        "push-left-front": ("push-left", "push-front"),
    }


class RQ3RepairMaximal(RQ3Repair):
    """RQ3, maximal: every training skill first, then one repair whose new literals are trained many times.

    The 13 training skills, which include every push but push-left-front, so (at-left ?o) and (at-front ?o) are each
    trained in several pushes and combined in the other three diagonals: the most favourable case for zero-shot
    transfer, and the longest rq1 runs. The controls and the metrics are RQ3Repair's (research_experiments.py).

    Pairs, each written with every training skill it relies on (relies_on):
      push-left, push-front, push-right-front, push-left-back, push-right-back -> push-left-front
          FROM is push-left; adds precondition (not (at-front ?o)) and effect (at-front ?o), in push-front and in
          the other diagonals, which also combine a side and a front or back region
    """

    name = "rq3-repair-maximal"
    base_skills = TRAINING_SKILLS
    relies_on = {"push-left-front": ("push-left", "push-front", "push-right-front", "push-left-back", "push-right-back")}
    repairs = (
        ("push-left", "push-left-front"),
        # TODO: enable stack -> place-beside, a pair with little overlap (stack and place-beside share few literals,
        # so it is closer to learning a new skill), once place-beside's carriers (putdown, push-beside) learn
        # reliably: until then it would only lengthen the plan without being informative.
        # ("stack", "place-beside"),
    )


class RepresentationOverlap(Experiment):
    """How much of their operators' structure do the training skills share, and how much of each held-out operator
    do they cover?

    Static: computed from the operators alone in under a second, nothing trains (src/cam/experiments/
    representation_overlap.py describes exactly what is counted). Each operator is split into units at five
    levels, from coarse to fine: the predicates it mentions, its atoms (predicate and arguments by parameter
    position), its literals (atom, place in the operator and sign: the multi-hot features and the compositional
    literal components), its whole precondition and effect, and the whole operator.

    The metrics that answer it, per level: the fraction of each skill's units that another training skill also has
    (skill shared; its mean over the skills) and the mean Jaccard similarity of pairs of skills (pair overlap); per
    held-out operator, the fraction of its units the training skills have (coverage). Which row to read depends on
    the embedding: multi-hot shares literals only, so the literals row is its similarity; the compositional
    encoders reuse predicates, atoms and literals at once, so for them the literals row is a lower bound (the
    report's "reused by" column says this per row); identifiers share nothing. They show that a policy could
    reuse structure, not that it does. The hypothesis they serve: the more of a skill's literals other training
    skills share, the more a structured embedding can help it over an identifier (RQ1); and a held-out operator can
    only be carried out zero-shot from structure seen in training, coverage 1 at the literal level (RQ2, RQ3).
    Every experiment prints the same report for its own skills when it starts, to read next to its results:

        .venv/bin/python scripts/overnight_runs.py --experiment overlap
        BASE_SKILLS="pickup unstack pickup-raised" REPAIRS="unstack:unstack-raised" \\
            .venv/bin/python scripts/overnight_runs.py --experiment overlap
    """

    name = "overlap"
    run_groups = ()
    base_skills = TRAINING_SKILLS
    repairs = RQ3RepairMinimal.repairs  # their TO's coverage


EXPERIMENTS: dict[str, type[Experiment]] = {
    experiment.name: experiment for experiment in (Learnability, RQ1, RQ3RepairMinimal, RQ3RepairMaximal, RepresentationOverlap)
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
