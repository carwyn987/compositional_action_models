"""scripts/review_runs.py: 95% CIs via bootstrap resampling over seeds, seeds that never reached the threshold, and the
per-condition report lines."""

import json

import numpy as np
import pytest

from scripts.review_runs import bootstrap_interval, mean_std, section_added_skills, section_conditions


def run(path, points, summary, retention=None):
    path.mkdir(parents=True)
    (path / "metrics.json").write_text(json.dumps({"curves": {"deterministic": {"points": points}},
                                                   "summary": summary, "retention": retention}))
    return path


def points(skill, steps, added=0):
    """Evaluations of skill every 50k steps from added to steps, each with the training progress up to it."""
    return [{"step": step, "result": {skill: {}},
             "progress": {"steps": step, "episodes": step // 100, "skill_steps": {skill: step - added},
                          "skill_episodes": {skill: (step - added) // 100}}}
            for step in range(added, steps + 1, 50_000)]


def summary(skill, steps_to_threshold, **values):
    return {mode: {skill: {"steps_to_threshold": steps_to_threshold, "skill_steps_to_threshold": steps_to_threshold,
                           "episodes_to_threshold": steps_to_threshold and steps_to_threshold // 100, **values}}
            for mode in ("deterministic", "stochastic")}


@pytest.mark.unit
def test_the_bootstrap_interval_is_reproducible_and_holds_the_true_mean_about_95_percent_of_the_time():
    """Over 200 samples of 20 seeds from N(5, 1), the interval holds the true mean 5 in about 95% of them, and the
    same values always give the same interval. With 3 seeds the interval is just their range, which holds the mean
    far less often (in 3 of 4 samples for a symmetric distribution)."""
    rng = np.random.default_rng(1)
    samples = [rng.normal(5, 1, size=20) for _ in range(200)]
    assert 0.88 <= np.mean([low <= 5 <= high for low, high in map(bootstrap_interval, samples)]) <= 0.98
    assert bootstrap_interval(samples[0]) == bootstrap_interval(list(samples[0]))
    assert bootstrap_interval([3.0, 1.0, 2.0]) == (1.0, 3.0)
    small = [rng.normal(5, 1, size=3) for _ in range(200)]
    assert np.mean([low <= 5 <= high for low, high in map(bootstrap_interval, small)]) < 0.85


@pytest.mark.unit
def test_one_seed_or_identical_seeds_give_no_spread():
    """A single value is its own interval, and the report shows it alone (as for a single seed); identical values
    give a zero-width interval."""
    assert bootstrap_interval([0.7]) == (0.7, 0.7)
    assert bootstrap_interval([0.7, 0.7, 0.7]) == pytest.approx((0.7, 0.7))
    assert mean_std([0.7]) == "0.7" and mean_std([0.7, None]) == "0.7"
    assert mean_std([0.7, 0.7, 0.7]) == "0.7 ± 0 [95% CI via bootstrap: 0.7, 0.7]"
    assert mean_std([0.0, 1.0]) == "0.5 ± 0.71 [95% CI via bootstrap: 0, 1]"


@pytest.mark.unit
def test_a_seed_that_never_reached_the_threshold_counts_as_what_it_trained_and_makes_a_lower_bound():
    """One seed reached the threshold at 100k; two never did in their 400k and 600k. Left out (the control), they
    make the condition look as fast as its one success. Counted as what they trained, the mean (366667) and its
    interval are lower bounds, marked as such: had those seeds reached it later, both ends would only be higher."""
    assert mean_std([100_000, None, None]) == "1e+05"
    assert mean_std([100_000, None, None], trained=[500_000, 400_000, 600_000]) == \
        "≥3.67e+05 ± 2.5e+05 [95% CI via bootstrap: 1e+05, 6e+05] (2 never)"
    later = bootstrap_interval([100_000, 700_000, 900_000, 300_000])
    counted = bootstrap_interval([100_000, 400_000, 600_000, 300_000])
    assert counted[0] <= later[0] and counted[1] <= later[1]
    assert mean_std([None, None], trained=[400_000, 600_000]) == "never"
    assert mean_std([100_000, 200_000], trained=[500_000, 400_000]) == \
        "1.5e+05 ± 7.1e+04 [95% CI via bootstrap: 1e+05, 2e+05]"


@pytest.mark.unit
def test_the_conditions_report_an_interval_over_the_seeds_of_each_condition(tmp_path):
    """rq1 keeps each seed in its own folder (rq1/<condition>_seed<n>/<run name>_seed<n>), so grouped by folder each
    seed would be a condition of its own. One-hot's two seeds are one condition with an interval, its seed that never
    reached the threshold counted as its 500k steps; multi-hot's single seed is a condition without one."""
    runs = [
        run(tmp_path / "rq1" / "one-hot_seed0" / "pickup_mtsac_one-hot_shaped_seed0", points("pickup", 500_000),
            summary("pickup", 100_000, auc=0.8, final_success_rate=1.0)),
        run(tmp_path / "rq1" / "one-hot_seed1" / "pickup_mtsac_one-hot_shaped_seed1", points("pickup", 500_000),
            summary("pickup", None, auc=0.4, final_success_rate=0.6)),
        run(tmp_path / "rq1" / "multi-hot_seed0" / "pickup_mtsac_multi-hot_shaped_seed0", points("pickup", 500_000),
            summary("pickup", 50_000, auc=0.9, final_success_rate=1.0)),
    ]
    lines = [" ".join(line.split()) for line in section_conditions(runs)]
    assert [line for line in lines[1:] if "seed" in line] == ["multi-hot/pickup_mtsac_multi-hot_shaped (1 seeds)",
                                                              "one-hot/pickup_mtsac_one-hot_shaped (2 seeds)"]
    one_hot = lines[lines.index("one-hot/pickup_mtsac_one-hot_shaped (2 seeds)") + 1]
    assert one_hot == ("deter pickup "
                       "steps ≥3e+05 ± 2.8e+05 [95% CI via bootstrap: 1e+05, 5e+05] (1 never) "
                       "on skill ≥3e+05 ± 2.8e+05 [95% CI via bootstrap: 1e+05, 5e+05] (1 never) "
                       "episodes ≥3e+03 ± 2.8e+03 [95% CI via bootstrap: 1e+03, 5e+03] (1 never) "
                       "auc 0.6 ± 0.28 [95% CI via bootstrap: 0.4, 0.8] "
                       "final 0.8 ± 0.28 [95% CI via bootstrap: 0.6, 1]")
    multi_hot = lines[lines.index("multi-hot/pickup_mtsac_multi-hot_shaped (1 seeds)") + 1]
    assert multi_hot == "deter pickup steps 5e+04 on skill 5e+04 episodes 500 auc 0.9 final 1"


@pytest.mark.unit
def test_added_skills_report_intervals_and_count_seeds_that_never_got_there(tmp_path):
    """pickup-raised added at 500k in two seeds of a repair: 20k and 40k steps to threshold. The other skills were
    back 30k steps after the addition in one seed and never in the other, which trained 100k steps after it. From
    scratch, one learnability seed needed 60k steps and the other never got there in 100k: a transfer gain of at
    least (60k + 100k) / 2 / 30k."""
    runs = [run(tmp_path / "learnability" / "pickup-raised" / f"pickup-raised_mtsac_multi-hot_shaped_seed{seed}",
                points("pickup-raised", 100_000), summary("pickup-raised", steps))
            for seed, steps in enumerate([60_000, None])]
    for seed, (steps, back) in enumerate([(20_000, 530_000), (40_000, None)]):
        retention = {"added_at": 500_000, "all_recovered_at": back,
                     "skills": {"pickup": {"level_before": 1.0, "lowest_after": 0.5, "final": 1.0}}}
        runs.append(run(tmp_path / "repair" / "pickup-raised" / f"one-hot_repair-visible_isolated_seed{seed}",
                        [{"step": 0, "result": {"pickup": {}}}] + [
                            {**point, "result": {"pickup": {}, "pickup-raised": {}}}
                            for point in points("pickup-raised", 600_000, added=500_000)],
                        summary("pickup-raised", steps, zero_shot_success_rate=0.1 * seed, final_success_rate=1.0),
                        retention))
    lines = [" ".join(line.split()) for line in section_added_skills(runs)]
    assert ("pickup-raised/one-hot_repair-visible_isolated pickup-raised (2 seeds; added at step 5e+05 ± 0 "
            "[95% CI via bootstrap: 5e+05, 5e+05])") in lines
    assert ("deter zero-shot 0.05 ± 0.071 [95% CI via bootstrap: 0, 0.1] "
            "steps to threshold 3e+04 ± 1.4e+04 [95% CI via bootstrap: 2e+04, 4e+04] "
            "on skill 3e+04 ± 1.4e+04 [95% CI via bootstrap: 2e+04, 4e+04] "
            "final 1 ± 0 [95% CI via bootstrap: 1, 1]") in lines
    assert ("from scratch (learnability) steps to threshold ≥8e+04 ± 2.8e+04 "
            "[95% CI via bootstrap: 6e+04, 1e+05] (1 never) transfer gain "
            "≥2.67x (from scratch / after the repair)") in lines
    assert any(line.endswith("steps until all back ≥6.5e+04 ± 4.9e+04 [95% CI via bootstrap: 3e+04, 1e+05] (1 never)")
               for line in lines)
