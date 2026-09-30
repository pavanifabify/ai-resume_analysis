"""
resume_parser.py
-----------------
Step 1 & 2 of the pipeline: extract raw text from PDF/DOCX, then parse it into
structured fields.

Design choice: contact info (email, phone, LinkedIn, GitHub) is extracted with
REGEX, not the LLM. Regex is deterministic and near-100% reliable for well-formed
patterns, whereas asking an LLM to "read" an email address risks silent typos.
Everything that requires judgment (skills grouping, project descriptions, etc.)
goes through the LLM with a strict JSON schema, and we merge/backfill the two.
"""

import io
import re
from typing import Optional

import pymupdf as fitz  # PyMuPDF (using the non-deprecated import alias)
import docx  # python-docx

from src.llm_client import LLMClient
from src.prompts import RESUME_PARSE_PROMPT
from src.config import Config


# ---------------------------------------------------------------------------
# Text extraction
# ---------------------------------------------------------------------------

def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract plain text from a PDF file's bytes using PyMuPDF."""
    try:
        text_parts = []
        with fitz.open(stream=file_bytes, filetype="pdf") as pdf:
            for page in pdf:
                text_parts.append(page.get_text("text"))
        return "\n".join(text_parts).strip()
    except Exception as e:
        raise ValueError(
            "This PDF could not be opened. It may be corrupted, password-protected, "
            "or not a valid PDF file."
        ) from e


def extract_text_from_docx(file_bytes: bytes) -> str:
    """Extract plain text from a DOCX file's bytes using python-docx."""
    try:
        document = docx.Document(io.BytesIO(file_bytes))
    except Exception as e:
        raise ValueError(
            "This DOCX file could not be opened. It may be corrupted or not a valid "
            "Word document."
        ) from e

    parts = [p.text for p in document.paragraphs if p.text.strip()]

    # Also pull text out of tables, since many resumes use table layouts.
    for table in document.tables:
        for row in table.rows:
            for cell in row.cells:
                if cell.text.strip():
                    parts.append(cell.text.strip())

    return "\n".join(parts).strip()


def extract_text(file_bytes: bytes, filename: str) -> str:
    """Dispatch to the correct extractor based on file extension."""
    ext = filename.lower().rsplit(".", 1)[-1]
    if ext == "pdf":
        text = extract_text_from_pdf(file_bytes)
    elif ext in ("docx",):
        text = extract_text_from_docx(file_bytes)
    else:
        raise ValueError(f"Unsupported file type: .{ext}. Please upload a PDF or DOCX.")

    if not text or len(text.strip()) < 30:
        raise ValueError(
            "Could not extract meaningful text from this file. It may be a scanned "
            "image-based PDF (no selectable text) or corrupted. Please upload a "
            "text-based PDF or DOCX."
        )

    return text[: Config.MAX_RESUME_CHARS]


# ---------------------------------------------------------------------------
# Deterministic regex extraction (contact info) - used as ground truth / backfill
# ---------------------------------------------------------------------------

EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")

# Matches phone numbers in common groupings: "+91 98765 43210", "(555) 123-4567",
# "555-123-4567", "9876543210", etc. Requires a leading boundary so it doesn't
# grab digits out of unrelated numbers (years, scores), and a minimum total digit
# count so short numbers (e.g. a CGPA "8.7") never match.
PHONE_RE = re.compile(
    r"(?<!\d)(\+\d{1,3}[\s\-.]?)?\(?\d{2,5}\)?[\s\-.]?\d{2,5}[\s\-.]?\d{2,5}(?:[\s\-.]?\d{2,4})?(?!\d)"
)

LINKEDIN_RE = re.compile(r"(https?://)?(www\.)?linkedin\.com/[a-zA-Z0-9\-_/]+", re.IGNORECASE)
GITHUB_RE = re.compile(r"(https?://)?(www\.)?github\.com/[a-zA-Z0-9\-_/]+", re.IGNORECASE)
PORTFOLIO_HINT_RE = re.compile(
    r"(?<![\w.\-])(https?://)?(www\.)?[a-zA-Z0-9\-]+(?:\.[a-zA-Z0-9\-]+)*\.(dev|me|io|com|in|tech|xyz|app)(/[a-zA-Z0-9\-_/]*)?",
    re.IGNORECASE,
)


def _looks_like_valid_phone(candidate: str) -> bool:
    """Reject matches that are too short (e.g. stray digit groups) to be a real phone number."""
    digit_count = sum(c.isdigit() for c in candidate)
    return 7 <= digit_count <= 15


def regex_extract_contact_info(text: str) -> dict:
    """Deterministically pull contact details via regex. High precision by design."""
    email_match = EMAIL_RE.search(text)
    linkedin_match = LINKEDIN_RE.search(text)
    github_match = GITHUB_RE.search(text)

    phone = None
    for match in PHONE_RE.finditer(text):
        candidate = match.group(0).strip()
        if _looks_like_valid_phone(candidate):
            phone = candidate
            break

    portfolio = None
    email_span = email_match.span() if email_match else None
    for match in PORTFOLIO_HINT_RE.finditer(text):
        candidate = match.group(0)
        low = candidate.lower()
        if "linkedin.com" in low or "github.com" in low:
            continue
        # Skip this match if it falls inside (or overlaps) the email address span,
        # since email local-parts/domains often look like a bare domain (e.g.
        # "ananya.rao.dev@gmail.com" contains a false-positive-looking "rao.dev").
        if email_span and match.start() < email_span[1] and match.end() > email_span[0]:
            continue
        portfolio = candidate
        break

    return {
        "email": email_match.group(0) if email_match else None,
        "phone": phone,
        "linkedin": linkedin_match.group(0) if linkedin_match else None,
        "github": github_match.group(0) if github_match else None,
        "portfolio": portfolio,
    }


# ---------------------------------------------------------------------------
# LLM-assisted structured parsing
# ---------------------------------------------------------------------------

def parse_resume_structured(resume_text: str, llm: Optional[LLMClient] = None) -> dict:
    """
    Parse the resume into the full structured schema (education, skills, experience,
    projects, certifications, achievements) using the LLM, then backfill/override
    contact info with the deterministic regex results for reliability.
    """
    llm = llm or LLMClient()

    prompt = RESUME_PARSE_PROMPT.format(resume_text=resume_text)
    system_prompt = (
        "You are a precise resume-parsing engine. You extract only information that "
        "is explicitly present in the text. You never fabricate details."
    )

    structured = llm.chat_json(system_prompt, prompt, temperature=0.0, max_tokens=2500)

    # Backfill with regex-extracted contact info (regex is more reliable for this).
    regex_contacts = regex_extract_contact_info(resume_text)
    personal = structured.setdefault("personal_info", {})
    for key, value in regex_contacts.items():
        if value and not personal.get(key):
            personal[key] = value

    # Defensive defaults so downstream code never KeyErrors on a sparse resume.
    structured.setdefault("education", [])
    structured.setdefault(
        "skills",
        {
            "programming_languages": [], "frameworks": [], "databases": [],
            "tools": [], "cloud": [], "testing": [],
        },
    )
    structured.setdefault("experience", [])
    structured.setdefault("projects", [])
    structured.setdefault("certifications", [])
    structured.setdefault("achievements", [])

    return structured
