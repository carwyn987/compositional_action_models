"""main.py run lifecycle: new run, resume with an added skill, evaluate only, locked options."""

import json
import shutil

import pytest

pytest.importorskip("stable_baselines3")

from stable_baselines3 import SAC  # noqa: E402

from main import main, parse_args, run_name  # noqa: E402

FAST = ["--algorithm", "sac", "--num-blocks", "2", "--max-steps-per-episode", "10", "--eval-episodes", "1"]


def run_directory(tmp_path, skills):
    return tmp_path / run_name(parse_args(["--skills", *skills, *FAST]))


@pytest.mark.integration
def test_new_run_then_resume_with_added_skill_then_evaluate(tmp_path):
    """Train pickup 200 steps; resume adding putdown for 100 more (its zero-shot point is the resume step);
    then evaluate only. Curves, step count, monitor.csv and train.log continue in the same directory."""
    main(["--skills", "pickup", *FAST, "--total-timesteps", "200", "--eval-interval", "100", "--output-dir", str(tmp_path)])
    run = run_directory(tmp_path, ["pickup"])
    main(["--run-directory", str(run), "--skills", "pickup", "putdown", "--total-timesteps", "100"])

    assert SAC.load(run / "model.zip").num_timesteps == 300
    metrics = json.loads((run / "metrics.json").read_text())
    points = metrics["curves"]["deterministic"]["points"]
    assert [point["step"] for point in points] == [0, 100, 200, 300]
    assert "putdown" not in points[1]["result"] and "putdown" in points[2]["result"]  # added at step 200
    assert metrics["progress"]["steps"] == 300
    assert json.loads((run / "config.json").read_text())["skills"] == ["pickup", "putdown"]
    assert (run / "train.log").read_text().count("to resume or continue this run") == 2

    main(["--mode", "evaluate", "--run-directory", str(run), "--eval-skills", "putdown"])
    (evaluation_file,) = run.glob("evaluation_*.json")
    evaluation = json.loads(evaluation_file.read_text())
    assert evaluation["step"] == 300 and set(evaluation["results"]["deterministic"]) == {"putdown"}


@pytest.mark.integration
def test_repair_modes_choose_which_skills_train_after_the_addition(tmp_path):
    """Train pickup 200 steps, then resume for 100 more with putdown added as the repaired skill, three ways.
    Control (no repair mode): the two take turns. isolated: only putdown trains, pickup is still evaluated, and
    its retention is recorded. retain: both train, taking turns as without a mode."""
    main(["--skills", "pickup", *FAST, "--total-timesteps", "200", "--eval-interval", "100", "--output-dir", str(tmp_path)])
    base = run_directory(tmp_path, ["pickup"])
    before = json.loads((base / "metrics.json").read_text())["progress"]["skill_episodes"]["pickup"]
    added = {}
    for mode in ("none", "isolated", "retain"):
        run = shutil.copytree(base, tmp_path / mode)
        options = [] if mode == "none" else ["--stop-skills", "putdown", "--repair-mode", mode]
        main(["--run-directory", str(run), "--skills", "pickup", "putdown", "--total-timesteps", "100", *options])
        metrics = json.loads((run / "metrics.json").read_text())
        episodes = metrics["progress"]["skill_episodes"]
        added[mode] = (episodes["pickup"] - before, episodes.get("putdown", 0))
        assert "pickup" in metrics["curves"]["deterministic"]["points"][-1]["result"]  # evaluated in every mode
        assert set(metrics["retention"]["skills"]) == {"pickup"}

    assert abs(added["none"][0] - added["none"][1]) <= 1
    assert added["isolated"][0] == 0 and added["isolated"][1] > 0
    assert abs(added["retain"][0] - added["retain"][1]) <= 1


@pytest.mark.unit
def test_repair_mode_needs_stop_skills():
    """--repair-mode says how the added skills train, so it needs them named in --stop-skills."""
    with pytest.raises(SystemExit):
        parse_args(["--skills", "pickup", "putdown", "--repair-mode", "isolated"])


@pytest.mark.integration
def test_skill_sampling_option_reaches_the_skill_environment():
    """--skill-sampling competence turns on SkillEnvironment's competence sampling; the default takes turns."""
    from cam.experiments.environment_setup import setup_environment
    from cam.skills.registry import build_skill

    for option, competence in (([], False), (["--skill-sampling", "competence"], True)):
        config = parse_args(["--skills", "pickup", "putdown", *FAST, *option])
        env = setup_environment(config, [build_skill(name, config) for name in config["skills"]])
        try:
            assert env.get_wrapper_attr("competence_sampling") is competence
        finally:
            env.close()


@pytest.mark.unit
def test_resume_rejects_options_that_change_the_observation(tmp_path):
    """Options defining the observation or algorithm (e.g. --operator-encoder) cannot change on resume."""
    (tmp_path / "config.json").write_text(json.dumps(parse_args(["--skills", "pickup"])))
    with pytest.raises(SystemExit):
        parse_args(["--run-directory", str(tmp_path), "--operator-encoder", "one-hot"])
    assert parse_args(["--run-directory", str(tmp_path), "--total-timesteps", "5"])["total_timesteps"] == 5


@pytest.mark.unit
def test_resume_keeps_saved_options_that_differ_from_the_defaults(tmp_path):
    """A run saved with non-default options (e.g. --algorithm ppo) resumes with them when they are not
    given again; only options typed on the command line override the saved config."""
    (tmp_path / "config.json").write_text(json.dumps(parse_args(["--skills", "pickup", "--algorithm", "ppo", "--seed", "3"])))
    config = parse_args(["--run-directory", str(tmp_path), "--skills", "pickup", "putdown"])
    assert (config["algorithm"], config["seed"], config["skills"]) == ("ppo", 3, ["pickup", "putdown"])


@pytest.mark.unit
def test_a_text_run_saved_without_operator_text_name_resumes_with_the_name_in_its_text(tmp_path):
    """The config.json of a text run saved before --operator-text-name has no such key. Resumed (with the default
    given explicitly too) or read as it is (as scripts/review_runs.py does), it embeds the operator's text with the
    name, as the run trained. Control: leaving the name out would change the observation, so it is refused."""
    import numpy as np

    from cam.domain.pddl.pddl_writer import to_pddl
    from cam.experiments.environment_setup import OPERATOR_ENCODERS
    from cam.representations.text_backends import MockTextBackend
    from cam.skills.registry import build_skill

    saved = parse_args(["--skills", "pickup", "--operator-encoder", "text"])
    del saved["operator_text_name"]
    (tmp_path / "config.json").write_text(json.dumps(saved))
    pickup = build_skill("pickup", saved).symbolic_action_model
    trained = MockTextBackend(saved["operator_embedding_dim"]).embed_text(to_pddl(pickup))  # "(:action pickup ..."
    for config in (parse_args(["--run-directory", str(tmp_path)]),
                   parse_args(["--run-directory", str(tmp_path), "--operator-text-name", "include"]), saved):
        np.testing.assert_array_equal(OPERATOR_ENCODERS["text"](config, []).encode(pickup), trained)
    with pytest.raises(SystemExit):
        parse_args(["--run-directory", str(tmp_path), "--operator-text-name", "none"])


@pytest.mark.unit
def test_a_compositional_run_saved_without_composition_normalization_resumes_with_the_output_normalized(tmp_path):
    """A config.json saved before --composition-normalization has no such key: resumed, the run keeps its only
    normalization at the output, as it trained; a new run normalizes every stage. Control: asking a resumed old run
    for every stage would change its network, so it is refused."""
    from main import policy_kwargs

    saved = parse_args(["--skills", "pickup", "--operator-encoder", "compositional"])
    assert policy_kwargs(saved)["features_extractor_kwargs"]["architecture_kwargs"] == {"stage_normalization": True}
    del saved["composition_normalization"]
    (tmp_path / "config.json").write_text(json.dumps(saved))
    resumed = parse_args(["--run-directory", str(tmp_path)])
    assert resumed["composition_normalization"] == "output"
    assert policy_kwargs(resumed)["features_extractor_kwargs"]["architecture_kwargs"] == {"stage_normalization": False}
    with pytest.raises(SystemExit):
        parse_args(["--run-directory", str(tmp_path), "--composition-normalization", "every-stage"])


@pytest.mark.unit
def test_a_run_saved_without_operator_embedding_loss_resumes_with_the_actor_loss(tmp_path):
    """The config.json of a run saved before --operator-embedding-loss has no such key; its policy, Stable-Baselines3's
    own, trained the shared operator embedding with the actor loss. Resumed (with actor given explicitly too), it
    keeps the actor loss, and the critic loss is refused. Control: a new run, and a saved run that has the key, use
    the critic loss."""
    saved = parse_args(["--skills", "pickup", "--trainable-operator-embedding"])
    assert saved["operator_embedding_loss"] == "critic"
    (tmp_path / "config.json").write_text(json.dumps(saved))
    assert parse_args(["--run-directory", str(tmp_path)])["operator_embedding_loss"] == "critic"
    del saved["operator_embedding_loss"]
    (tmp_path / "config.json").write_text(json.dumps(saved))
    for explicit in ([], ["--operator-embedding-loss", "actor"]):
        assert parse_args(["--run-directory", str(tmp_path), *explicit])["operator_embedding_loss"] == "actor"
    with pytest.raises(SystemExit):
        parse_args(["--run-directory", str(tmp_path), "--operator-embedding-loss", "critic"])
