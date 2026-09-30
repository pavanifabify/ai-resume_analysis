"""
candidate_classifier.py
------------------------
Step 3 of the pipeline: classify the candidate into one of six profile types.

Design choice: we compute cheap, deterministic RULE-BASED signals first (years of
experience, graduation year vs current year, internship count, project count).
These are passed to the LLM as strong hints. The LLM makes the final call because
it can read nuance (e.g. "Career Switcher" requires understanding that experience
is in an unrelated field), but grounding it in rule-based hints substantially
reduces misclassification versus asking the LLM to reason from scratch.
"""

import json
import re
from datetime import datetime
from typing import Optional

from src.llm_client import LLMClient
from src.prompts import CANDIDATE_CLASSIFICATION_PROMPT

VALID_TYPES = {
    "Student", "Fresher", "Internship Candidate",
    "Entry Level Professional", "Experienced Professional", "Career Switcher",
}


def _extract_years(text: str):
    """Pull 4-digit years out of a string (handles '2021', '2021-2023', 'Present', etc.)."""
    return [int(y) for y in re.findall(r"\b(19|20)\d{2}\b", text or "")]


def compute_rule_based_hints(structured: dict) -> dict:
    """Compute deterministic signals used to ground the LLM's classification."""
    current_year = datetime.now().year

    education = structured.get("education", []) or []
    experience = structured.get("experience", []) or []
    projects = structured.get("projects", []) or []
    certifications = structured.get("certifications", []) or []

    # Graduation year: take the max year mentioned across education entries.
    grad_years = []
    for edu in education:
        grad_years.extend(_extract_years(str(edu.get("graduation_year", ""))))
    latest_grad_year = max(grad_years) if grad_years else None

    is_current_or_future_grad = (
        latest_grad_year is not None and latest_grad_year >= current_year
    )
    graduated_recently = (
        latest_grad_year is not None and 0 <= (current_year - latest_grad_year) <= 1
    )

    # Rough full-time experience count: entries whose role doesn't scream "intern".
    internship_count = 0
    full_time_count = 0
    for exp in experience:
        role = (exp.get("role") or "").lower()
        if "intern" in role:
            internship_count += 1
        else:
            full_time_count += 1

    return {
        "current_year": current_year,
        "latest_graduation_year": latest_grad_year,
        "is_currently_studying_or_future_grad": is_current_or_future_grad,
        "graduated_within_last_year": graduated_recently,
        "internship_count": internship_count,
        "full_time_experience_entries": full_time_count,
        "project_count": len(projects),
        "certification_count": len(certifications),
    }


def classify_candidate(structured: dict, llm: Optional[LLMClient] = None) -> dict:
    """
    Classify the candidate. Returns:
    {"candidate_type": ..., "confidence": ..., "reason": ...}
    Falls back to a pure rule-based classification if the LLM call fails.
    """
    llm = llm or LLMClient()
    hints = compute_rule_based_hints(structured)

    prompt = CANDIDATE_CLASSIFICATION_PROMPT.format(
        structured_data=json.dumps(structured, indent=2),
        rule_based_hints=json.dumps(hints, indent=2),
    )
    system_prompt = (
        "You are an expert technical recruiter who classifies candidates accurately "
        "based on evidence, not assumptions."
    )

    try:
        result = llm.chat_json(system_prompt, prompt, temperature=0.0, max_tokens=500)
        candidate_type = result.get("candidate_type", "").strip()
        if candidate_type not in VALID_TYPES:
            raise ValueError(f"LLM returned invalid candidate_type: {candidate_type}")
        result.setdefault("confidence", "Medium")
        result.setdefault("reason", "Classified based on resume content.")
        return result
    except Exception:
        return _rule_based_fallback(hints)


def _rule_based_fallback(hints: dict) -> dict:
    """A safe, deterministic fallback if the LLM classification fails entirely."""
    if hints["full_time_experience_entries"] >= 2:
        return {
            "candidate_type": "Experienced Professional",
            "confidence": "Low",
            "reason": "Fallback rule-based classification: 2+ full-time experience entries detected.",
        }
    if hints["full_time_experience_entries"] == 1:
        return {
            "candidate_type": "Entry Level Professional",
            "confidence": "Low",
            "reason": "Fallback rule-based classification: 1 full-time experience entry detected.",
        }
    if hints["internship_count"] >= 1 and hints["is_currently_studying_or_future_grad"]:
        return {
            "candidate_type": "Internship Candidate",
            "confidence": "Low",
            "reason": "Fallback rule-based classification: internship experience while still studying.",
        }
    if hints["graduated_within_last_year"]:
        return {
            "candidate_type": "Fresher",
            "confidence": "Low",
            "reason": "Fallback rule-based classification: graduated within the last year, no full-time experience.",
        }
    return {
        "candidate_type": "Student",
        "confidence": "Low",
        "reason": "Fallback rule-based classification: no full-time experience, likely still studying.",
    }
