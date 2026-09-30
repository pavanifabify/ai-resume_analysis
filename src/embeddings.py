"""
embeddings.py
-------------
Embedding wrapper around Sentence-Transformers.

The model is loaded once per process and reused through an LRU cache.
Embeddings are L2-normalized so FAISS inner-product search can be used
as cosine-similarity search.
"""

from functools import lru_cache

import numpy as np

from src.config import Config


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_embedding_model():
    """
    Load and cache the Sentence-Transformers model.

    The model is loaded only once per Python process.
    """

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        raise ImportError(
            "sentence-transformers is not installed. "
            "Install it with: pip install sentence-transformers"
        ) from e

    try:
        model = SentenceTransformer(
            Config.EMBEDDING_MODEL_NAME
        )
    except Exception as e:
        raise RuntimeError(
            f"Failed to load embedding model "
            f"'{Config.EMBEDDING_MODEL_NAME}'. "
            f"Check your sentence-transformers/transformers installation "
            f"and internet connection if the model has not been downloaded."
        ) from e

    return model


# ---------------------------------------------------------------------------
# Text embedding
# ---------------------------------------------------------------------------

def embed_texts(texts: list[str]) -> np.ndarray:
    """
    Convert multiple texts into normalized float32 embeddings.

    Returns:
        numpy.ndarray with shape:
        (number_of_texts, embedding_dimension)
    """

    if not texts:
        return np.empty(
            (0, 0),
            dtype=np.float32,
        )

    # Convert everything to strings.
    cleaned_texts = [
        str(text).strip()
        for text in texts
    ]

    model = get_embedding_model()

    try:
        vectors = model.encode(
            cleaned_texts,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
    except Exception as e:
        raise RuntimeError(
            "Failed to generate text embeddings."
        ) from e

    vectors = np.asarray(
        vectors,
        dtype=np.float32,
    )

    # Make sure a single embedding is still 2-dimensional.
    if vectors.ndim == 1:
        vectors = vectors.reshape(
            1,
            -1,
        )

    return vectors


# ---------------------------------------------------------------------------
# Query embedding
# ---------------------------------------------------------------------------

def embed_query(text: str) -> np.ndarray:
    """
    Convert one query into a normalized embedding.

    Returns:
        numpy.ndarray with shape (1, embedding_dimension)
    """

    if not text or not text.strip():
        raise ValueError(
            "Query text cannot be empty."
        )

    return embed_texts(
        [text]
    )