"""Local semantic embedding provider using Sentence Transformers.

Optional: importing ``sentence_transformers`` (and by extension torch)
happens lazily inside ``__init__``, never at module import time, so this
module can always be imported by the base package. The model is downloaded
and loaded once, at construction, only when this provider is explicitly
requested. Nothing here happens at package import time.

Default model: ``sentence-transformers/all-MiniLM-L6-v2``. Chosen for a
small download and low CPU cost on a local machine, English engineering
text quality, a stable, widely used checkpoint, and a practical 384-dim
vector for local SQLite vector search, not for benchmark prestige. The
evaluation harness compares retrieval strategies (lexical, vector, hybrid)
against this one configured model; ranking this model against alternative
embedding models is a separate exercise it does not perform.
"""

from __future__ import annotations

from collections.abc import Sequence

from engineering_knowledge.embeddings.base import (
    EmbeddingConfigurationError,
    validate_embedding_vector,
)

DEFAULT_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"


class SentenceTransformersEmbeddingProvider:
    """Local embedding provider backed by a Sentence Transformers model.

    Uses the model's dedicated ``encode_query``/``encode_document`` methods
    rather than the generic ``encode``, so a model defining distinct
    query/document prompts is honored automatically; the default model has
    no such prompts configured, so both currently behave like plain
    encoding, but the intended API is still the correct one to call.
    """

    provider_type = "sentence_transformers"
    model_revision: str | None = None

    def __init__(self, *, model_id: str = DEFAULT_MODEL_ID, device: str | None = None) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as error:
            raise EmbeddingConfigurationError(
                "sentence-transformers is required for this provider; install the "
                "'local-embeddings' optional dependency"
            ) from error

        try:
            # Model construction can fail for reasons this boundary has no
            # useful way to enumerate (missing/renamed model, no network,
            # a corrupted local cache, an incompatible checkpoint format):
            # every one of them is a configuration problem from here, never
            # a bug in this constructor, which does nothing else. This is
            # a deliberately narrow exception to catching only concrete
            # types, scoped to exactly this third-party call.
            self._model = SentenceTransformer(model_id, device=device)
        except Exception as error:
            raise EmbeddingConfigurationError(f"failed to load model {model_id!r}") from error

        dimension = self._model.get_embedding_dimension()
        if dimension is None:
            raise EmbeddingConfigurationError(
                f"model {model_id!r} does not report an embedding dimension"
            )

        self.model_id = model_id
        self.dimension = dimension

    def embed_documents(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]:
        if not texts:
            return ()
        vectors = self._model.encode_document(list(texts), convert_to_numpy=True)
        return tuple(
            validate_embedding_vector(vector.tolist(), expected_dimension=self.dimension)
            for vector in vectors
        )

    def embed_query(self, text: str) -> tuple[float, ...]:
        vector = self._model.encode_query(text, convert_to_numpy=True)
        return validate_embedding_vector(vector.tolist(), expected_dimension=self.dimension)
