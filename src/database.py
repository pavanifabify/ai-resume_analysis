"""
database.py
------------
SQLite persistence layer for the AI Resume Analyzer.

Tables:
- users              : one row per known user
- resume_history     : one row per uploaded resume
- analysis_reports   : one row per full analysis run
- job_descriptions   : one row per job description used for matching

All functions use short-lived SQLite connections, which works well
with Streamlit's rerun model.
"""

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from typing import Optional

from src.config import Config


# ---------------------------------------------------------------------------
# Connection
# ---------------------------------------------------------------------------

@contextmanager
def get_connection():
    """
    Open a SQLite connection, enable foreign-key enforcement,
    commit successful operations, and always close the connection.
    """

    conn = sqlite3.connect(
        Config.DATABASE_PATH
    )

    conn.row_factory = sqlite3.Row

    # SQLite does not enforce foreign keys by default.
    conn.execute(
        "PRAGMA foreign_keys = ON"
    )

    try:
        yield conn
        conn.commit()

    except Exception:
        conn.rollback()
        raise

    finally:
        conn.close()


def _now() -> str:
    """
    Return the current local time as an ISO-8601 string.
    """
    return datetime.now().astimezone().isoformat()


# ---------------------------------------------------------------------------
# Database initialization
# ---------------------------------------------------------------------------

def init_db():
    """
    Create all required tables if they do not already exist.

    Safe to call every time the Streamlit application starts.
    """

    with get_connection() as conn:

        cur = conn.cursor()

        # ---------------------------------------------------------------
        # Users
        # ---------------------------------------------------------------

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                email TEXT UNIQUE,
                created_at TEXT NOT NULL
            )
            """
        )

        # ---------------------------------------------------------------
        # Resume history
        # ---------------------------------------------------------------

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS resume_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                resume_filename TEXT NOT NULL,
                upload_date TEXT NOT NULL,
                raw_text TEXT,

                FOREIGN KEY (user_id)
                    REFERENCES users(id)
                    ON DELETE SET NULL
            )
            """
        )

        # ---------------------------------------------------------------
        # Analysis reports
        # ---------------------------------------------------------------

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS analysis_reports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                resume_history_id INTEGER NOT NULL,
                candidate_type TEXT,
                overall_score INTEGER,
                report_json TEXT NOT NULL,
                created_at TEXT NOT NULL,

                FOREIGN KEY (resume_history_id)
                    REFERENCES resume_history(id)
                    ON DELETE CASCADE
            )
            """
        )

        # ---------------------------------------------------------------
        # Job descriptions
        # ---------------------------------------------------------------

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS job_descriptions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                resume_history_id INTEGER,
                jd_text TEXT NOT NULL,
                match_result_json TEXT,
                created_at TEXT NOT NULL,

                FOREIGN KEY (resume_history_id)
                    REFERENCES resume_history(id)
                    ON DELETE CASCADE
            )
            """
        )


# ---------------------------------------------------------------------------
# Users
# ---------------------------------------------------------------------------

def get_or_create_user(
    name: Optional[str],
    email: Optional[str],
) -> Optional[int]:
    """
    Return a user_id for the supplied email.

    If the email does not exist, create a new user.

    If no email is available, return None so anonymous
    resume analysis is still supported.
    """

    if not email:
        return None

    email = email.strip().lower()

    with get_connection() as conn:

        cur = conn.cursor()

        # Check whether the user already exists.
        cur.execute(
            """
            SELECT id
            FROM users
            WHERE email = ?
            """,
            (email,),
        )

        row = cur.fetchone()

        if row:
            return row["id"]

        # Create user.
        cur.execute(
            """
            INSERT INTO users (
                name,
                email,
                created_at
            )
            VALUES (?, ?, ?)
            """,
            (
                name,
                email,
                _now(),
            ),
        )

        return cur.lastrowid


# ---------------------------------------------------------------------------
# Resume history
# ---------------------------------------------------------------------------

def save_resume_history(
    user_id: Optional[int],
    filename: str,
    raw_text: str,
) -> int:
    """
    Save an uploaded resume and return its database ID.
    """

    with get_connection() as conn:

        cur = conn.cursor()

        cur.execute(
            """
            INSERT INTO resume_history (
                user_id,
                resume_filename,
                upload_date,
                raw_text
            )
            VALUES (?, ?, ?, ?)
            """,
            (
                user_id,
                filename,
                _now(),
                raw_text,
            ),
        )

        return cur.lastrowid


def get_resume_history(
    user_id: Optional[int] = None,
) -> list:
    """
    Return resume history.

    If user_id is supplied, only that user's resumes are returned.
    Otherwise, all resumes are returned.
    """

    with get_connection() as conn:

        cur = conn.cursor()

        if user_id is not None:

            cur.execute(
                """
                SELECT *
                FROM resume_history
                WHERE user_id = ?
                ORDER BY upload_date DESC
                """,
                (user_id,),
            )

        else:

            cur.execute(
                """
                SELECT *
                FROM resume_history
                ORDER BY upload_date DESC
                """
            )

        return [
            dict(row)
            for row in cur.fetchall()
        ]


# ---------------------------------------------------------------------------
# Analysis reports
# ---------------------------------------------------------------------------

def save_analysis_report(
    resume_history_id: int,
    candidate_type: str,
    overall_score: int,
    report: dict,
) -> int:
    """
    Save a complete analysis report as JSON.

    Returns the analysis report ID.
    """

    # Defensive score normalization.
    try:
        overall_score = int(
            round(
                float(
                    overall_score
                )
            )
        )
    except (
        TypeError,
        ValueError,
    ):
        overall_score = 0

    overall_score = max(
        0,
        min(
            100,
            overall_score,
        ),
    )

    with get_connection() as conn:

        cur = conn.cursor()

        cur.execute(
            """
            INSERT INTO analysis_reports (
                resume_history_id,
                candidate_type,
                overall_score,
                report_json,
                created_at
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                resume_history_id,
                candidate_type,
                overall_score,
                json.dumps(
                    report,
                    ensure_ascii=False,
                ),
                _now(),
            ),
        )

        return cur.lastrowid


def get_analysis_reports(
    resume_history_id: Optional[int] = None,
) -> list:
    """
    Retrieve saved analysis reports.

    The JSON report is automatically converted back
    into a Python dictionary.
    """

    with get_connection() as conn:

        cur = conn.cursor()

        if resume_history_id is not None:

            cur.execute(
                """
                SELECT *
                FROM analysis_reports
                WHERE resume_history_id = ?
                ORDER BY created_at DESC
                """,
                (resume_history_id,),
            )

        else:

            cur.execute(
                """
                SELECT *
                FROM analysis_reports
                ORDER BY created_at DESC
                """
            )

        rows = []

        for row in cur.fetchall():

            item = dict(row)

            try:
                item["report"] = json.loads(
                    item["report_json"]
                )

            except (
                json.JSONDecodeError,
                TypeError,
            ):
                item["report"] = {}

            rows.append(item)

        return rows


# ---------------------------------------------------------------------------
# Job descriptions
# ---------------------------------------------------------------------------

def save_job_description(
    resume_history_id: Optional[int],
    jd_text: str,
    match_result: Optional[dict] = None,
) -> int:
    """
    Save a job description and optional JD matching result.

    Returns the job-description record ID.
    """

    with get_connection() as conn:

        cur = conn.cursor()

        match_json = None

        if match_result is not None:

            match_json = json.dumps(
                match_result,
                ensure_ascii=False,
            )

        cur.execute(
            """
            INSERT INTO job_descriptions (
                resume_history_id,
                jd_text,
                match_result_json,
                created_at
            )
            VALUES (?, ?, ?, ?)
            """,
            (
                resume_history_id,
                jd_text,
                match_json,
                _now(),
            ),
        )

        return cur.lastrowid


def get_job_descriptions(
    resume_history_id: Optional[int] = None,
) -> list:
    """
    Retrieve saved job descriptions.

    If a match result exists, it is returned as a Python dictionary.
    """

    with get_connection() as conn:

        cur = conn.cursor()

        if resume_history_id is not None:

            cur.execute(
                """
                SELECT *
                FROM job_descriptions
                WHERE resume_history_id = ?
                ORDER BY created_at DESC
                """,
                (resume_history_id,),
            )

        else:

            cur.execute(
                """
                SELECT *
                FROM job_descriptions
                ORDER BY created_at DESC
                """
            )

        rows = []

        for row in cur.fetchall():

            item = dict(row)

            if item.get("match_result_json"):

                try:
                    item["match_result"] = json.loads(
                        item["match_result_json"]
                    )

                except (
                    json.JSONDecodeError,
                    TypeError,
                ):
                    item["match_result"] = {}

            else:
                item["match_result"] = None

            rows.append(item)

        return rows