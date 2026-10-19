"""cam.experiments.research_experiments: every repair asks only for literals the training skills have."""

import pytest

from cam.experiments.research_experiments import EXPERIMENTS, TRAINING_SKILLS, untrained_literals


@pytest.mark.unit
@pytest.mark.parametrize("experiment", EXPERIMENTS.values(), ids=EXPERIMENTS)
def test_every_repair_holds_out_an_operator_whose_literals_are_all_trained(experiment):
    """Each repair replaces a training skill with a held-out operator, and every literal of that operator (place in
    the operator, sign, predicate and arguments) appears in some training skill, in another context."""
    for repair_from, repair_to in experiment.repairs:
        assert repair_from in experiment.base_skills and repair_to not in experiment.base_skills
        assert untrained_literals(repair_to, experiment.base_skills) == []


@pytest.mark.unit
def test_a_literal_counts_as_trained_only_with_its_place_sign_and_arguments():
    """Control: without push-beside, place-beside's (beside ?o ?u) is untrained, and so is its precondition
    (on-table ?u), although (on-table ?o) is a precondition of pickup and an effect of putdown."""
    without_push_beside = [skill for skill in TRAINING_SKILLS if skill != "push-beside"]
    assert untrained_literals("place-beside", without_push_beside) == ["effect (beside ?o ?u)",
                                                                       "precondition (on-table ?u)"]
    assert untrained_literals("push-left-front", ["push-left"]) == ["effect (at-front ?o)",
                                                                    "precondition (not (at-front ?o))"]
    assert untrained_literals("push-left-front", ["push-left", "push-front"]) == []


@pytest.mark.unit
@pytest.mark.parametrize("name", ["rq3-repair-minimal", "rq3-repair-maximal"])
def test_each_repair_relies_on_training_skills_that_have_every_literal_of_the_repaired_operator(name):
    """The skills a repair is documented to rely on are training skills, start with FROM, and between them have every
    literal of TO; without them (the control, FROM alone) a literal is missing."""
    from cam.experiments.research_experiments import EXPERIMENTS, untrained_literals

    experiment = EXPERIMENTS[name]
    assert set(experiment.relies_on) == {to for _, to in experiment.repairs}
    for repair_from, repair_to in experiment.repairs:
        skills = experiment.relies_on[repair_to]
        assert skills[0] == repair_from and set(skills) <= set(experiment.base_skills)
        assert untrained_literals(repair_to, skills) == []
        assert untrained_literals(repair_to, [repair_from]) != []
