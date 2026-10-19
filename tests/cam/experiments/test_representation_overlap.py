"""The representation overlap of operators, on pairs whose overlap is known by hand."""

import pytest

from cam.experiments.representation_overlap import operator_units, overlap, report


def level(rows, name):
    return next(row for row in rows if row.level == name)


@pytest.mark.unit
def test_pickup_and_pickup_raised_differ_in_one_literal_and_one_section():
    """pickup deletes (raised ?o) where pickup-raised adds it: the same 5 predicates and atoms, 7 of 9 literals in
    common (each has 8, the multi-hot embedding's positional features), the same precondition but different
    effects, and different operators."""
    assert len(operator_units("pickup")["literals"]) == 8
    rows = overlap(["pickup", "pickup-raised"])
    assert level(rows, "predicates").pair_overlap == 1.0 and level(rows, "atoms").pair_overlap == 1.0
    literals = level(rows, "literals")
    assert literals.units == 9 and literals.pair_overlap == pytest.approx(7 / 9)
    assert literals.skill_shared == {"pickup": pytest.approx(7 / 8), "pickup-raised": pytest.approx(7 / 8)}
    sections = level(rows, "sections")
    assert sections.units == 3 and sections.skill_shared == {"pickup": 0.5, "pickup-raised": 0.5}
    assert level(rows, "operators").shared_units == 0.0


@pytest.mark.unit
def test_atoms_are_by_parameter_position_not_variable_name():
    """place-beside's second parameter ?u is position 1, wherever it appears."""
    assert "on-table(1)" in operator_units("place-beside")["atoms"]
    assert "add beside(0,1)" in operator_units("place-beside")["literals"]


@pytest.mark.unit
def test_coverage_counts_the_held_out_operators_units_seen_in_training():
    """unstack-raised is unstack with (raised ?o) added: unstack alone covers all but that literal; with pickup-raised,
    which adds (raised ?o) too, every literal is covered (what the overnight runner requires of a repair)."""
    alone = level(overlap(["unstack"], held_out=["unstack-raised"]), "literals").coverage["unstack-raised"]
    assert alone == pytest.approx(8 / 9)
    covered = level(overlap(["unstack", "pickup-raised"], held_out=["unstack-raised"]), "literals")
    assert covered.coverage["unstack-raised"] == 1.0
    text = report(["unstack", "pickup-raised"], ["unstack-raised"])
    assert "coverage unstack-raised" in text
    literals_row = next(line for line in text.splitlines() if line.strip().startswith("literals"))
    assert "multi-hot (its similarity)" in literals_row and "lower bound" in literals_row
