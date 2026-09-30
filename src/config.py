"""
config.py
---------
Central configuration for the AI Resume Analyzer.

Reads settings from environment variables loaded from .env.
"""

import os
from dotenv import load_dotenv


# ==============================================================
# LOAD ENVIRONMENT VARIABLES
# ==============================================================

load_dotenv()


class Config:

    # ==========================================================
    # LLM PROVIDER
    # ==========================================================

    # Supported providers:
    #   openai
    #   gemini

    LLM_PROVIDER = os.getenv(
        "LLM_PROVIDER",
        "gemini"
    ).lower()

    # ==========================================================
    # OPENAI
    # ==========================================================

    OPENAI_API_KEY = os.getenv(
        "OPENAI_API_KEY",
        ""
    )

    OPENAI_MODEL = os.getenv(
        "OPENAI_MODEL",
        "gpt-4o-mini"
    )

    # ==========================================================
    # GEMINI
    # ==========================================================

    GEMINI_API_KEY = os.getenv(
        "GEMINI_API_KEY",
        ""
    )

    GEMINI_MODEL = os.getenv(
        "GEMINI_MODEL",
        "gemini-2.5-flash"
    )

    # ==========================================================
    # EMBEDDINGS
    # ==========================================================

    EMBEDDING_MODEL_NAME = os.getenv(
        "EMBEDDING_MODEL_NAME",
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    # ==========================================================
    # PROJECT PATHS
    # ==========================================================

    BASE_DIR = os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )

    KNOWLEDGE_BASE_DIR = os.path.join(
        BASE_DIR,
        "knowledge_base"
    )

    VECTOR_STORE_DIR = os.path.join(
        BASE_DIR,
        "vector_store"
    )

    DATABASE_DIR = os.path.join(
        BASE_DIR,
        "database"
    )

    DATABASE_PATH = os.path.join(
        DATABASE_DIR,
        "resume_analyzer.db"
    )

    FAISS_INDEX_PATH = os.path.join(
        VECTOR_STORE_DIR,
        "knowledge_base.index"
    )

    FAISS_META_PATH = os.path.join(
        VECTOR_STORE_DIR,
        "knowledge_base_meta.json"
    )

    # ==========================================================
    # RAG SETTINGS
    # ==========================================================

    CHUNK_SIZE = int(
        os.getenv(
            "CHUNK_SIZE",
            "500"
        )
    )

    CHUNK_OVERLAP = int(
        os.getenv(
            "CHUNK_OVERLAP",
            "80"
        )
    )

    TOP_K_RETRIEVAL = int(
        os.getenv(
            "TOP_K_RETRIEVAL",
            "4"
        )
    )

    # ==========================================================
    # APPLICATION SETTINGS
    # ==========================================================

    MAX_RESUME_CHARS = int(
        os.getenv(
            "MAX_RESUME_CHARS",
            "20000"
        )
    )

    # ==========================================================
    # VALIDATION
    # ==========================================================

    @classmethod
    def validate(cls):
        """
        Validate the application configuration.

        Returns:
            list[str]: List of configuration problems.
        """

        problems = []

        # Check provider
        if cls.LLM_PROVIDER not in (
            "openai",
            "gemini"
        ):
            problems.append(
                "LLM_PROVIDER must be "
                "'openai' or 'gemini', "
                f"got '{cls.LLM_PROVIDER}'"
            )

        # Check OpenAI key
        if (
            cls.LLM_PROVIDER == "openai"
            and not cls.OPENAI_API_KEY
        ):
            problems.append(
                "OPENAI_API_KEY is not set "
                "(required because "
                "LLM_PROVIDER=openai)"
            )

        # Check Gemini key
        if (
            cls.LLM_PROVIDER == "gemini"
            and not cls.GEMINI_API_KEY
        ):
            problems.append(
                "GEMINI_API_KEY is not set "
                "(required because "
                "LLM_PROVIDER=gemini)"
            )

        return problems


# ==============================================================
# CREATE REQUIRED DIRECTORIES
# ==============================================================

os.makedirs(
    Config.VECTOR_STORE_DIR,
    exist_ok=True
)

os.makedirs(
    Config.DATABASE_DIR,
    exist_ok=True
)