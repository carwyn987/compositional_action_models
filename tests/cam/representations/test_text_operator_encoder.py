import dataclasses

import numpy as np
import pytest

from cam.domain.pddl.pddl import Precondition
from cam.domain.pddl.pddl_parser import parse_operator
from cam.domain.symbols import Predicate
from cam.representations.embedding_cache import DiskEmbeddingCache
from cam.representations.text_backends import CachedTextBackend, MockTextBackend, TextEmbeddingBackend
from cam.representations.text_operator_encoder import TextOperatorEncoder, operator_paragraph
from tests.cam.domain.conftest import LEGACY_OPERATORS

PICKUP = parse_operator(LEGACY_OPERATORS["pickup"])


@pytest.mark.unit
def test_mock_backend_is_deterministic_and_unit_norm():
    vector = MockTextBackend(128).embed_text("(:action pickup)")
    assert vector.shape == (128,) and np.linalg.norm(vector) == pytest.approx(1.0)
    np.testing.assert_array_equal(vector, MockTextBackend(128).embed_text("(:action pickup)"))


@pytest.mark.unit
def test_mock_backend_reflects_shared_substrings():
    """Texts sharing most of their n-grams are closer than unrelated texts."""
    backend = MockTextBackend(128)
    a, similar, unrelated = (backend.embed_text(t) for t in (
        "pickup when on-table and clear", "pickup when on-table and clear and dry", "zebra quokka xylophone"))
    assert a @ similar > 0.8 > a @ unrelated


@pytest.mark.unit
def test_paragraph_describes_preconditions_and_effects():
    paragraph = operator_paragraph(PICKUP)
    assert paragraph.startswith("Action pickup with parameters ?o (block).")
    assert "can be applied when (on-table ?o), (clear ?o) and (gripper-empty) hold" in paragraph
    assert "Afterwards (holding ?o) holds, and (on-table ?o) and (gripper-empty) no longer hold." in paragraph


@pytest.mark.unit
@pytest.mark.parametrize("text", ["pddl", "paragraph"])
def test_repaired_operator_gets_a_new_embedding(text):
    """The text, and so the embedding, follows the operator's content."""
    repaired = dataclasses.replace(
        PICKUP, preconditions=PICKUP.preconditions + (Precondition(Predicate("holding", ("?o",)), negated=True),)
    )
    encoder = TextOperatorEncoder(MockTextBackend(128), text)
    assert encoder.dim == 128
    assert not np.allclose(encoder.encode(PICKUP), encoder.encode(repaired))


@pytest.mark.unit
def test_cached_backend_embeds_each_text_once_and_persists(tmp_path):
    """A cached backend calls the wrapped backend once per text; a new cache on the same file reuses it."""

    class CountingBackend(TextEmbeddingBackend):
        id, dim, calls = "counting", 4, 0

        def embed_text(self, text):
            CountingBackend.calls += 1
            return np.full(4, len(text), dtype=np.float32)

    path = tmp_path / "cache.json"
    for _ in range(2):
        backend = CachedTextBackend(CountingBackend(), DiskEmbeddingCache(path))
        np.testing.assert_array_equal(backend.embed_text("abc"), [3, 3, 3, 3])
    assert CountingBackend.calls == 1
    assert not list(tmp_path.glob("*.partial"))


@pytest.mark.unit
def test_api_key_comes_from_the_environment_or_the_key_file(tmp_path, monkeypatch):
    """OPENAI_API_KEY wins; otherwise the stripped contents of OPENAI_API_KEY_FILE (a fake key here)."""
    from cam.representations.text_backends import openai_api_key

    key_file = tmp_path / "key"
    key_file.write_text("sk-test-fake\n")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY_FILE", str(key_file))
    assert openai_api_key() == "sk-test-fake"
    monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env")
    assert openai_api_key() == "sk-from-env"


@pytest.mark.unit
def test_missing_api_key_error_does_not_contain_a_key(tmp_path, monkeypatch):
    from cam.representations.text_backends import openai_api_key

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY_FILE", str(tmp_path / "missing"))
    with pytest.raises(RuntimeError, match="no OpenAI API key") as error:
        openai_api_key()
    assert "sk-" not in str(error.value)
