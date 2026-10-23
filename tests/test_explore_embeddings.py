"""scripts/explore_embeddings.py: runs are explored as their policy computes the embedding, the workspace keeps only
valid PDDL, the tables answer known cases, and the app serves the same chart."""

import json
import threading
import urllib.request

import gymnasium as gym
import numpy as np
import pytest

pytest.importorskip("stable_baselines3")
pytest.importorskip("plotly")

import torch  # noqa: E402
from stable_baselines3 import PPO, SAC  # noqa: E402

from cam.experiments.embedding_points import registered_operators  # noqa: E402
from cam.policies.feature_extractors import features_extractors  # noqa: E402
from main import parse_args as main_config, policy_kwargs  # noqa: E402
from scripts.explore_embeddings import (  # noqa: E402
    Explorer,
    add_point,
    build_sources,
    load_workspace,
    make_server,
    parse_args,
    save_workspace,
    update_workspace,
)

class OperatorEnv(gym.Env):
    """The policy observation's shape for an operator embedding of the given size, with nothing to learn."""

    action_space = gym.spaces.Box(-1, 1, (1,))

    def __init__(self, size: int):
        self.observation_space = gym.spaces.Dict({
            "observation": gym.spaces.Box(-1, 1, (1,)), "operator_embedding": gym.spaces.Box(-np.inf, np.inf, (size,)),
        })

    def reset(self, *, seed=None, options=None):
        return {key: np.zeros(space.shape, np.float32) for key, space in self.observation_space.items()}, {}

    def step(self, action):
        return self.reset()[0], 0.0, True, False, {}


def saved_run(folder, algorithm, *argv):
    """A run folder (config.json, model.zip) of an untrained policy whose learned embedding is moved away from its
    initialization, as training would (W doubled, the composition scaled)."""
    config = main_config(["--skills", "pickup", "putdown", "--algorithm", algorithm, *argv])
    from cam.experiments.environment_setup import OPERATOR_ENCODERS

    size = OPERATOR_ENCODERS[config["operator_encoder"]](config, []).dim
    options = {"n_steps": 64, "batch_size": 32} if algorithm == "ppo" else {}
    model = (PPO if algorithm == "ppo" else SAC)(
        "MultiInputPolicy", OperatorEnv(size), policy_kwargs=policy_kwargs(config) | {"net_arch": [8]}, device="cpu",
        **options)
    with torch.no_grad():
        for parameter in features_extractors(model.policy)["actor"].parameters():
            parameter.mul_(1.5)
    folder.mkdir()
    model.save(folder / "model.zip")
    (folder / "config.json").write_text(json.dumps(config))
    return model


def explorer(tmp_path, *argv) -> Explorer:
    return Explorer(parse_args(["--workspace", str(tmp_path / "workspace.json"), "--untrained-seeds", "1", *argv]))


@pytest.mark.unit
@pytest.mark.parametrize("algorithm, encoder", [
    ("sac", ["--operator-encoder", "multi-hot", "--trainable-operator-embedding"]),
    ("ppo", ["--operator-encoder", "multi-hot", "--trainable-operator-embedding"]),
    ("sac", ["--operator-encoder", "compositional", "--compositional-name", "none"]),
])
def test_a_runs_points_are_what_its_policy_computes(tmp_path, algorithm, encoder):
    """Each source's operator points are the extractor's operator_embedding of the encoder's vector, for every
    extractor of the run (--extractor all): the critic shares the actor's, so it is not a source of its own."""
    run = tmp_path / "run"
    model = saved_run(run, algorithm, *encoder)
    sources, twins, notes = build_sources(parse_args(["--run", str(run), "--extractor", "all", "--untrained-seeds", "0"]))
    extractors = features_extractors(model.policy)
    assert [source.name for source in sources] == [f"run {role}" for role in extractors]
    assert "critic" not in extractors  # main.py's policies share the actor's extractor with the critic
    operators = registered_operators()
    for source, extractor in zip(sources, extractors.values()):
        points, unencodable = source.points("operator", operators)
        assert not unencodable
        inputs = torch.as_tensor(np.stack([source.encoder.encode(entry.operator) for entry in operators]))
        with torch.no_grad():
            expected = extractor.operator_embedding(inputs).numpy()
        np.testing.assert_allclose(np.stack([point.vector for point in points]), expected, rtol=1e-5, atol=1e-6)
        assert {point.label: point.trained for point in points}["pickup"] is True
        assert {point.label: point.trained for point in points}["stack"] is False
    if encoder[1] == "compositional":
        (twin,) = twins["run actor"]
        learned = sources[0].points("operator", operators)[0][0].vector
        untrained = twin.points("operator", operators)[0][0].vector
        again = build_sources(parse_args(["--run", str(run), "--untrained-seeds", "0"]))[1]["run actor"][0]
        np.testing.assert_array_equal(untrained, again.points("operator", operators)[0][0].vector)
        assert not np.allclose(learned, untrained)


def rows(result: dict, title: str) -> list[dict]:
    """A chart's table as {column: cell} per row."""
    table = next(table for table in result["tables"] if table.title == title)
    return [dict(zip(table.columns, row)) for row in table.rows]


@pytest.mark.unit
def test_tables_answer_known_cases(tmp_path):
    """multi-hot: pickup and pickup-raised have cosine 0.875 and their analogy with unstack is exact; structure
    agreement is high for multi-hot, about 0 for random and n/a for one-hot (every pair has cosine 0)."""
    result = explorer(tmp_path, "--static", "one-hot", "random", "multi-hot").chart({})
    agreement = {row["source"]: row["Spearman"] for row in rows(result, "structure agreement")}
    assert agreement["one-hot"] == "n/a"
    assert float(agreement["multi-hot"]) > 0.5 > abs(float(agreement["random"]))
    pair = next(row for row in rows(result, "pairs") if (row["A"], row["B"], row["source"]) == ("pickup", "pickup-raised", "multi-hot"))
    assert pair["cosine"] == "0.875" and pair["rank"] == "1 / 1"
    analogy = next(row for row in rows(result, "analogies")
                   if (row["analogy"], row["source"]) == ("pickup:pickup-raised::unstack:unstack-raised", "multi-hot"))
    assert analogy["direction cosine"] == "1.000" and analogy["rank of D"].startswith("1 of")
    assert set(result["figures"]) == {"2"} and set(result["figures"]["2"]) == {"light", "dark"}


@pytest.mark.unit
def test_the_workspace_keeps_valid_pddl_only(tmp_path):
    """A custom point round-trips through the workspace file; invalid PDDL and a label in use are refused with the
    reason, and nothing is saved."""
    path = tmp_path / "workspace.json"
    workspace = load_workspace(path)
    assert ["pickup", "pickup-raised"] in workspace["pairs"]  # a new workspace starts with the default questions
    add_point(workspace, "raised copy", "(:action pickup-raised :parameters (?o - block) :precondition (and) "
                                        ":effect (and (raised ?o)))", "note")
    save_workspace(path, workspace)
    for label, pddl in (("broken", "(:action broken :parameters (?o - block)"), ("raised copy", "(:action x :parameters () "
                        ":precondition (and) :effect (and))"), ("pickup", "(:action x :parameters () :precondition (and) :effect (and))")):
        with pytest.raises(ValueError):
            add_point(workspace, label, pddl)
    assert load_workspace(path) == json.loads(path.read_text())
    assert [point["label"] for point in load_workspace(path)["custom_points"]] == ["raised copy"]


@pytest.mark.unit
def test_custom_points_and_expressions_in_the_chart(tmp_path):
    """A custom point and an expression are placed in every source that can encode them, and listed with their
    nearest operators; a compositional source refuses the expression (its literals are in another space)."""
    app = explorer(tmp_path, "--static", "multi-hot", "compositional-tree", "--expression",
                   "raised?=pickup - eff (not (raised ?o)) + eff (raised ?o)")
    update_workspace(app, app.args)
    status, body = app.api_add_point({"label": "copy", "pddl": "(:action pickup-raised :parameters (?x - block) "
                                      ":precondition (and (on-table ?x) (clear ?x) (gripper-empty)) :effect (and "
                                      "(holding ?x) (raised ?x) (not (on-table ?x)) (not (clear ?x)) (not (gripper-empty))))"})
    assert status == 200 and body["errors"] == {}
    result = app.chart({})
    nearest = {(row["point"], row["source"]): row["nearest"] for row in rows(result, "custom and expression points")}
    assert nearest[("raised?", "multi-hot")].startswith("pickup-raised 1.000")
    assert nearest[("copy", "multi-hot")].startswith("pickup-raised 1.000")
    refused = rows(result, "not encodable")
    assert [row["label"] for row in refused] == ["raised?"] and "different spaces" in refused[0]["reason"]


@pytest.mark.unit
def test_the_app_serves_the_page_and_saves_a_point(tmp_path):
    """End to end on a free port: the page, the options, a chart, and a point posted and saved to the workspace."""
    app = explorer(tmp_path, "--static", "multi-hot")
    server = make_server(app, 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_address[1]}"

    def call(method, path, body=None):
        request = urllib.request.Request(url + path, method=method, data=None if body is None else json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request) as response:
                return response.status, response.read()
        except urllib.error.HTTPError as error:
            return error.code, error.read()

    try:
        status, page = call("GET", "/")
        assert status == 200 and b"/plotly.js" in page and b"Operator embeddings" in page
        status, options = call("GET", "/api/options")
        assert json.loads(options)["sources"][0]["name"] == "multi-hot"
        status, chart = call("POST", "/api/chart", {"levels": ["operator", "literal"], "dims": [3]})
        assert status == 200 and set(json.loads(chart)["figures"]) == {"3"}
        status, error = call("POST", "/api/points", {"label": "bad", "pddl": "(:action"})
        assert status == 400 and "error" in json.loads(error)
        status, _ = call("POST", "/api/points", {"label": "mine", "pddl": "(:action pickup :parameters (?o - block) "
                                                 ":precondition (and) :effect (and (holding ?o)))"})
        assert status == 200
        assert [point["label"] for point in json.loads((tmp_path / "workspace.json").read_text())["custom_points"]] == ["mine"]
        status, _ = call("DELETE", "/api/points/mine")
        assert status == 200 and json.loads((tmp_path / "workspace.json").read_text())["custom_points"] == []
    finally:
        server.shutdown()
        server.server_close()


@pytest.mark.unit
def test_export_writes_the_chart_and_its_data(tmp_path):
    """The export: a page with both themes' figures for each --dims, the points, the vectors and the tables."""
    app = explorer(tmp_path, "--static", "multi-hot", "--dims", "2", "3", "--plotlyjs", "cdn")
    app.export({}, tmp_path / "out", "cdn")
    page = (tmp_path / "out" / "embeddings.html").read_text()
    data = json.loads(page.split("const DATA = ")[1].split(";\nconst state")[0])
    assert set(data["figures"]) == {"2", "3"} and "cdn.plot.ly" in page
    vectors = np.load(tmp_path / "out" / "vectors.npz")
    assert vectors["multi-hot | operator"].shape == (16, 256)
    assert (tmp_path / "out" / "points.csv").read_text().count("\n") == 17
    assert "PAIRS" in (tmp_path / "out" / "tables.txt").read_text()
    assert "relative distance" in (tmp_path / "out" / "pairs.csv").read_text()
