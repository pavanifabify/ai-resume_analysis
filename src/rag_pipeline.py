"""
rag_pipeline.py
----------------
Step 10 of the AI Resume Analyzer pipeline.

Pipeline:

Knowledge Base
      ↓
Text Extraction
      ↓
Chunking
      ↓
Sentence Transformers
      ↓
FAISS Vector Database
      ↓
Similarity Search
      ↓
Relevant Context
      ↓
LLM
      ↓
Career / Resume Answer

The FAISS index and metadata are cached on disk so that the knowledge base
does not need to be embedded every time the application starts.
"""

import glob
import json
import os
from typing import List, Optional

import numpy as np

from src.config import Config
from src.embeddings import embed_texts, embed_query
from src.llm_client import LLMClient
from src.prompts import (
    RAG_ANSWER_PROMPT,
    CAREER_ROADMAP_PROMPT,
)


# ---------------------------------------------------------------------------
# FAISS
# ---------------------------------------------------------------------------

def _get_faiss():
    """
    Import FAISS lazily.

    This prevents the whole application from crashing during import if
    FAISS is not installed. Instead, a clear error is produced when RAG
    functionality is actually used.
    """

    try:
        import faiss
        return faiss

    except ImportError as e:
        raise ImportError(
            "FAISS is required for the RAG pipeline.\n"
            "Install it with:\n"
            "pip install faiss-cpu"
        ) from e


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def chunk_text(
    text: str,
    chunk_size: Optional[int] = None,
    overlap: Optional[int] = None,
) -> List[str]:
    """
    Split text into overlapping chunks.

    Paragraph boundaries are preferred where possible.

    Args:
        text:
            Input document text.

        chunk_size:
            Maximum target chunk size in characters.

        overlap:
            Number of characters carried from the previous chunk.

    Returns:
        List of text chunks.
    """

    if not text or not text.strip():
        return []

    chunk_size = (
        Config.CHUNK_SIZE
        if chunk_size is None
        else chunk_size
    )

    overlap = (
        Config.CHUNK_OVERLAP
        if overlap is None
        else overlap
    )

    # Defensive validation.
    if chunk_size <= 0:
        raise ValueError(
            "CHUNK_SIZE must be greater than 0."
        )

    if overlap < 0:
        raise ValueError(
            "CHUNK_OVERLAP cannot be negative."
        )

    if overlap >= chunk_size:
        raise ValueError(
            "CHUNK_OVERLAP must be smaller than CHUNK_SIZE."
        )

    paragraphs = [
        paragraph.strip()
        for paragraph in text.split("\n")
        if paragraph.strip()
    ]

    chunks: List[str] = []
    current = ""

    for paragraph in paragraphs:

        # Paragraph fits into current chunk.
        if (
            current
            and len(current) + len(paragraph) + 1 <= chunk_size
        ):
            current = (
                current
                + "\n"
                + paragraph
            ).strip()

        elif not current:
            # Start first chunk.
            current = paragraph

        else:
            # Store current chunk.
            chunks.append(current)

            # Carry over the end of previous chunk.
            tail = (
                current[-overlap:]
                if overlap > 0
                else ""
            )

            current = (
                f"{tail}\n{paragraph}"
                if tail
                else paragraph
            ).strip()

    if current:
        chunks.append(current)

    # -----------------------------------------------------------------------
    # Handle paragraphs/chunks that are significantly larger than chunk_size.
    # -----------------------------------------------------------------------

    final_chunks: List[str] = []

    step = chunk_size - overlap

    for chunk in chunks:

        if len(chunk) <= chunk_size:
            final_chunks.append(chunk)
            continue

        for start in range(
            0,
            len(chunk),
            step,
        ):
            piece = chunk[
                start:start + chunk_size
            ]

            if piece.strip():
                final_chunks.append(
                    piece.strip()
                )

    return final_chunks


# ---------------------------------------------------------------------------
# Knowledge-base loading
# ---------------------------------------------------------------------------

def _load_knowledge_base_files() -> List[dict]:
    """
    Load .txt, .md and .pdf files from the knowledge_base directory.

    Returns:
        [
            {
                "source": "filename.pdf",
                "text": "..."
            }
        ]
    """

    docs: List[dict] = []

    if not os.path.isdir(
        Config.KNOWLEDGE_BASE_DIR
    ):
        return docs

    patterns = (
        "*.txt",
        "*.md",
        "*.pdf",
    )

    for pattern in patterns:

        paths = sorted(
            glob.glob(
                os.path.join(
                    Config.KNOWLEDGE_BASE_DIR,
                    pattern,
                )
            )
        )

        for path in paths:

            try:

                # -----------------------------------------------------------
                # PDF
                # -----------------------------------------------------------

                if path.lower().endswith(".pdf"):

                    try:
                        import pymupdf as fitz
                    except ImportError:
                        import fitz

                    with fitz.open(path) as pdf:

                        text = "\n".join(
                            page.get_text("text")
                            for page in pdf
                        )

                # -----------------------------------------------------------
                # TXT / Markdown
                # -----------------------------------------------------------

                else:

                    with open(
                        path,
                        "r",
                        encoding="utf-8",
                        errors="ignore",
                    ) as file:

                        text = file.read()

                if text and text.strip():

                    docs.append(
                        {
                            "source": os.path.basename(path),
                            "text": text.strip(),
                        }
                    )

            except Exception as e:

                # Do not allow one bad knowledge-base file to
                # destroy the entire RAG pipeline.
                print(
                    f"Warning: Could not load knowledge-base file "
                    f"'{path}': {e}"
                )

    return docs


# ---------------------------------------------------------------------------
# Index building
# ---------------------------------------------------------------------------

def build_index(
    force_rebuild: bool = False,
) -> None:
    """
    Build the FAISS index from the knowledge base.

    If a valid cached index already exists, rebuilding is skipped unless
    force_rebuild=True.
    """

    faiss = _get_faiss()

    index_path = Config.FAISS_INDEX_PATH
    meta_path = Config.FAISS_META_PATH

    # -----------------------------------------------------------------------
    # Use cached index if available.
    # -----------------------------------------------------------------------

    if (
        not force_rebuild
        and os.path.exists(index_path)
        and os.path.exists(meta_path)
    ):
        return

    # -----------------------------------------------------------------------
    # Load documents.
    # -----------------------------------------------------------------------

    docs = _load_knowledge_base_files()

    if not docs:

        raise FileNotFoundError(
            "No knowledge-base documents were found.\n\n"
            f"Expected files inside:\n"
            f"{Config.KNOWLEDGE_BASE_DIR}\n\n"
            "Supported formats:\n"
            "- .txt\n"
            "- .md\n"
            "- .pdf"
        )

    # -----------------------------------------------------------------------
    # Chunk documents.
    # -----------------------------------------------------------------------

    all_chunks: List[str] = []
    metadata: List[dict] = []

    for doc in docs:

        chunks = chunk_text(
            doc["text"]
        )

        for chunk in chunks:

            if not chunk.strip():
                continue

            all_chunks.append(
                chunk
            )

            metadata.append(
                {
                    "source": doc["source"],
                    "text": chunk,
                }
            )

    if not all_chunks:

        raise ValueError(
            "Knowledge-base documents were found, "
            "but no usable text chunks could be created."
        )

    # -----------------------------------------------------------------------
    # Generate embeddings.
    # -----------------------------------------------------------------------

    vectors = embed_texts(
        all_chunks
    )

    if vectors.size == 0:

        raise ValueError(
            "Embedding generation returned no vectors."
        )

    if vectors.ndim != 2:

        raise ValueError(
            f"Expected a 2D embedding matrix, "
            f"got shape {vectors.shape}."
        )

    if vectors.shape[0] != len(all_chunks):

        raise ValueError(
            "Number of embeddings does not match "
            "number of text chunks."
        )

    # FAISS expects float32.
    vectors = np.asarray(
        vectors,
        dtype=np.float32,
    )

    dimension = vectors.shape[1]

    if dimension <= 0:

        raise ValueError(
            "Embedding dimension is invalid."
        )

    # -----------------------------------------------------------------------
    # Create FAISS index.
    #
    # Because embeddings are normalized in embeddings.py,
    # inner product = cosine similarity.
    # -----------------------------------------------------------------------

    index = faiss.IndexFlatIP(
        dimension
    )

    index.add(
        vectors
    )

    # -----------------------------------------------------------------------
    # Make sure directories exist.
    # -----------------------------------------------------------------------

    os.makedirs(
        os.path.dirname(index_path),
        exist_ok=True,
    )

    # -----------------------------------------------------------------------
    # Save index.
    # -----------------------------------------------------------------------

    faiss.write_index(
        index,
        index_path,
    )

    with open(
        meta_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metadata,
            file,
            ensure_ascii=False,
            indent=2,
        )


# ---------------------------------------------------------------------------
# Index loading
# ---------------------------------------------------------------------------

def _load_index():
    """
    Load FAISS index and metadata.

    If they do not exist, build them first.
    """

    faiss = _get_faiss()

    index_path = Config.FAISS_INDEX_PATH
    meta_path = Config.FAISS_META_PATH

    # -----------------------------------------------------------------------
    # Build if missing.
    # -----------------------------------------------------------------------

    if (
        not os.path.exists(index_path)
        or not os.path.exists(meta_path)
    ):
        build_index()

    # -----------------------------------------------------------------------
    # Load FAISS.
    # -----------------------------------------------------------------------

    try:

        index = faiss.read_index(
            index_path
        )

    except Exception as e:

        raise RuntimeError(
            "The FAISS index could not be loaded. "
            "The index may be corrupted or incompatible with the installed "
            "FAISS version. Delete the vector_store files and rebuild."
        ) from e

    # -----------------------------------------------------------------------
    # Load metadata.
    # -----------------------------------------------------------------------

    try:

        with open(
            meta_path,
            "r",
            encoding="utf-8",
        ) as file:

            metadata = json.load(file)

    except Exception as e:

        raise RuntimeError(
            "The FAISS metadata file could not be loaded. "
            "Delete the vector_store files and rebuild the index."
        ) from e

    # -----------------------------------------------------------------------
    # Validate metadata.
    # -----------------------------------------------------------------------

    if not isinstance(
        metadata,
        list,
    ):

        raise ValueError(
            "FAISS metadata must contain a list."
        )

    if index.ntotal != len(metadata):

        raise ValueError(
            "FAISS index and metadata are out of sync.\n"
            f"Index vectors: {index.ntotal}\n"
            f"Metadata entries: {len(metadata)}\n\n"
            "Delete the vector_store directory and rebuild the index."
        )

    return index, metadata


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def retrieve(
    query: str,
    top_k: Optional[int] = None,
) -> List[dict]:
    """
    Retrieve the most relevant knowledge-base chunks.

    Returns:
        [
            {
                "source": "...",
                "text": "...",
                "score": 0.87
            }
        ]
    """

    if not query or not query.strip():
        return []

    top_k = (
        Config.TOP_K_RETRIEVAL
        if top_k is None
        else top_k
    )

    if top_k <= 0:
        return []

    index, metadata = _load_index()

    if index.ntotal == 0:
        return []

    # Never request more vectors than actually exist.
    search_k = min(
        top_k,
        index.ntotal,
    )

    query_vector = embed_query(
        query
    )

    if query_vector.ndim != 2:

        raise ValueError(
            f"Expected query embedding to be 2D, "
            f"got shape {query_vector.shape}."
        )

    scores, indices = index.search(
        query_vector,
        search_k,
    )

    results: List[dict] = []

    for score, index_id in zip(
        scores[0],
        indices[0],
    ):

        if index_id < 0:
            continue

        if index_id >= len(metadata):
            continue

        item = dict(
            metadata[index_id]
        )

        item["score"] = float(
            score
        )

        results.append(
            item
        )

    return results


# ---------------------------------------------------------------------------
# RAG answer generation
# ---------------------------------------------------------------------------

def rag_answer(
    query: str,
    candidate_summary: str = "",
    llm: Optional[LLMClient] = None,
    top_k: Optional[int] = None,
) -> dict:
    """
    Retrieve relevant knowledge and generate a grounded answer.
    """

    if not query or not query.strip():

        raise ValueError(
            "Query cannot be empty."
        )

    llm = llm or LLMClient()

    chunks = retrieve(
        query,
        top_k=top_k,
    )

    if chunks:

        context = "\n\n---\n\n".join(
            f"[{chunk['source']}]\n{chunk['text']}"
            for chunk in chunks
        )

    else:

        context = (
            "No relevant knowledge-base context was found."
        )

    prompt = RAG_ANSWER_PROMPT.format(
        context=context,
        candidate_summary=candidate_summary,
        query=query,
    )

    system_prompt = (
        "You are a career coach and resume expert. "
        "Ground your advice in the provided context. "
        "Do not invent facts from the knowledge base. "
        "If the context is insufficient, clearly say so. "
        "Give concrete and actionable advice."
    )

    answer = llm.chat(
        system_prompt,
        prompt,
        temperature=0.3,
        max_tokens=1200,
    )

    return {
        "answer": answer,
        "sources": [
            chunk["source"]
            for chunk in chunks
        ],
    }


# ---------------------------------------------------------------------------
# Career roadmap
# ---------------------------------------------------------------------------

def generate_career_roadmap(
    candidate_type: str,
    target_role: str,
    missing_skills: list,
    llm: Optional[LLMClient] = None,
) -> dict:
    """
    Retrieve career-roadmap information and generate a structured roadmap.
    """

    llm = llm or LLMClient()

    query = (
        "career roadmap and skill development plan "
        f"for a {candidate_type} "
        f"targeting {target_role}"
    )

    chunks = retrieve(
        query,
        top_k=4,
    )

    if chunks:

        context = "\n\n---\n\n".join(
            f"[{chunk['source']}]\n{chunk['text']}"
            for chunk in chunks
        )

    else:

        context = (
            "No relevant career-roadmap context was found."
        )

    if missing_skills:

        missing_skills_text = ", ".join(
            str(skill)
            for skill in missing_skills
        )

    else:

        missing_skills_text = "None identified"

    prompt = CAREER_ROADMAP_PROMPT.format(
        context=context,
        candidate_type=candidate_type,
        target_role=target_role,
        missing_skills=missing_skills_text,
    )

    system_prompt = (
        "You are a career coach. "
        "Return valid JSON only. "
        "Match the requested schema exactly. "
        "Use the retrieved context as grounding."
    )

    result = llm.chat_json(
        system_prompt,
        prompt,
        temperature=0.3,
        max_tokens=1200,
    )

    # -----------------------------------------------------------------------
    # Defensive defaults.
    # -----------------------------------------------------------------------

    result.setdefault(
        "short_term_0_3_months",
        [],
    )

    result.setdefault(
        "mid_term_3_9_months",
        [],
    )

    result.setdefault(
        "long_term_9_18_months",
        [],
    )

    result.setdefault(
        "recommended_resources",
        [],
    )

    return result