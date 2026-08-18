import math
import sys

import pytest

from engineering_knowledge.domain import SectionPath
from engineering_knowledge.embeddings import fingerprint as fingerprint_module
from engineering_knowledge.embeddings.base import EmbeddingError, validate_embedding_vector
from engineering_knowledge.embeddings.fake import FakeEmbeddingProvider
from engineering_knowledge.embeddings.fingerprint import (
    build_embedding_text,
    derive_embedding_fingerprint,
)


def test_derive_embedding_fingerprint_deterministic_and_sensitive_to_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline = derive_embedding_fingerprint(
        provider_type="fake", model_id="m1", model_revision=None, dimension=8
    )
    repeat = derive_embedding_fingerprint(
        provider_type="fake", model_id="m1", model_revision=None, dimension=8
    )
    assert baseline == repeat

    assert baseline != derive_embedding_fingerprint(
        provider_type="other", model_id="m1", model_revision=None, dimension=8
    )
    assert baseline != derive_embedding_fingerprint(
        provider_type="fake", model_id="m2", model_revision=None, dimension=8
    )
    assert baseline != derive_embedding_fingerprint(
        provider_type="fake", model_id="m1", model_revision="rev1", dimension=8
    )
    assert baseline != derive_embedding_fingerprint(
        provider_type="fake", model_id="m1", model_revision=None, dimension=16
    )

    # the embedding-text representation version participates too: bumping
    # it must invalidate every previously computed fingerprint even when
    # the provider/model description is otherwise identical
    monkeypatch.setattr(fingerprint_module, "EMBEDDING_TEXT_VERSION", "different-version")
    assert baseline != fingerprint_module.derive_embedding_fingerprint(
        provider_type="fake", model_id="m1", model_revision=None, dimension=8
    )


def test_build_embedding_text_composes_title_section_and_chunk_text() -> None:
    root_text = build_embedding_text(
        title="runbook.md", section_path=SectionPath(), chunk_text="intro"
    )
    assert root_text == "runbook.md\nintro"

    sectioned_text = build_embedding_text(
        title="runbook.md",
        section_path=SectionPath(headings=("Payments", "Rollback")),
        chunk_text="steps",
    )
    assert sectioned_text == "runbook.md\nPayments Rollback\nsteps"


@pytest.mark.parametrize(
    "vector",
    [
        (1.0, 2.0),
        (1.0, float("nan"), 3.0),
        (1.0, float("inf"), 3.0),
    ],
)
def test_validate_embedding_vector_rejects_invalid_output(vector: tuple[float, ...]) -> None:
    with pytest.raises(EmbeddingError):
        validate_embedding_vector(vector, expected_dimension=3)


def test_fake_embedding_provider_is_deterministic_and_dimension_valid() -> None:
    provider = FakeEmbeddingProvider(dimension=6)

    first = provider.embed_query("rollback steps")
    second = provider.embed_query("rollback steps")
    assert first == second
    assert len(first) == 6
    assert all(math.isfinite(value) for value in first)

    documents = provider.embed_documents(["a", "b"])
    assert len(documents) == 2
    assert documents[0] != documents[1]


def test_sentence_transformers_provider_reports_configuration_error_without_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from engineering_knowledge.embeddings.base import EmbeddingConfigurationError
    from engineering_knowledge.embeddings.sentence_transformers_provider import (
        SentenceTransformersEmbeddingProvider,
    )

    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    with pytest.raises(EmbeddingConfigurationError):
        SentenceTransformersEmbeddingProvider()
