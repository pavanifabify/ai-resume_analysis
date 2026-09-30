"""
analyzer.py
-----------
Steps 5-9 of the pipeline:

5. Resume scoring
6. Skill analysis
7. Project evaluation
8. Experience evaluation
9. Job description matching

Every function accepts plain dicts/strings and returns plain dictionaries.
"""

import json
from typing import Optional

from src.llm_client import LLMClient
from src.prompts import (
    SCORING_PROMPT,
    SKILL_ANALYSIS_PROMPT,
    PROJECT_EVALUATION_PROMPT,
    EXPERIENCE_EVALUATION_PROMPT,
    JD_MATCH_PROMPT,
    COMBINED_RESUME_ANALYSIS_PROMPT,
)


# ==============================================================
# COMMON SYSTEM PROMPT
# ==============================================================

RECRUITER_SYSTEM_PROMPT = (
    "You are a senior technical recruiter. "
    "Give specific, evidence-based and honest feedback. "
    "Calibrate all judgments to the candidate's career stage. "
    "Never invent experience, skills, projects, achievements, "
    "metrics, technologies, or qualifications. "
    "Return valid JSON matching the requested structure."
)


# ==============================================================
# HELPERS
# ==============================================================

def _clamp_score(
    value,
    low: int = 0,
    high: int = 100,
) -> int:
    """
    Convert a value to an integer score and keep it inside
    the requested range.
    """

    try:
        value = int(round(float(value)))
    except (
        TypeError,
        ValueError,
    ):
        return low

    return max(
        low,
        min(high, value),
    )


def _ensure_list(value):
    """Return value as a list, or an empty list if invalid."""

    if isinstance(value, list):
        return value

    return []


# ==============================================================
# STEP 5: RESUME SCORING
# ==============================================================

def score_resume(
    structured: dict,
    resume_text: str,
    candidate_type: str,
    llm: Optional[LLMClient] = None,
) -> dict:

    llm = llm or LLMClient()

    prompt = SCORING_PROMPT.format(
        candidate_type=candidate_type,
        structured_data=json.dumps(
            structured,
            indent=2,
        ),
        resume_text=resume_text[:6000],
    )

    result = llm.chat_json(
        RECRUITER_SYSTEM_PROMPT,
        prompt,
        temperature=0.1,
        max_tokens=1800,
    )

    # ----------------------------------------------------------
    # Overall score
    # ----------------------------------------------------------

    result["overall_score"] = _clamp_score(
        result.get(
            "overall_score",
            0,
        )
    )

    # ----------------------------------------------------------
    # Category scores
    # ----------------------------------------------------------

    category_scores = result.get(
        "category_scores",
        {},
    )

    if not isinstance(
        category_scores,
        dict,
    ):
        category_scores = {}

    score_categories = (
        "ats_compatibility",
        "formatting",
        "technical_skills",
        "projects",
        "experience",
        "education",
        "achievements",
        "grammar",
    )

    for key in score_categories:

        category_scores[key] = _clamp_score(
            category_scores.get(
                key,
                0,
            )
        )

    result["category_scores"] = category_scores

    # ----------------------------------------------------------
    # Strengths / weaknesses
    # ----------------------------------------------------------

    result["strengths"] = _ensure_list(
        result.get(
            "strengths",
            [],
        )
    )

    result["weaknesses"] = _ensure_list(
        result.get(
            "weaknesses",
            [],
        )
    )

    return result


# ==============================================================
# STEP 6: SKILL ANALYSIS
# ==============================================================

def analyze_skills(
    structured: dict,
    candidate_type: str,
    llm: Optional[LLMClient] = None,
) -> dict:

    llm = llm or LLMClient()

    projects_and_experience = {
        "projects": structured.get(
            "projects",
            [],
        ),
        "experience": structured.get(
            "experience",
            [],
        ),
    }

    prompt = SKILL_ANALYSIS_PROMPT.format(
        candidate_type=candidate_type,
        skills_json=json.dumps(
            structured.get(
                "skills",
                {},
            ),
            indent=2,
        ),
        projects_and_experience=json.dumps(
            projects_and_experience,
            indent=2,
        ),
    )

    result = llm.chat_json(
        RECRUITER_SYSTEM_PROMPT,
        prompt,
        temperature=0.2,
        max_tokens=1500,
    )

    result.setdefault(
        "inferred_target_role",
        "General Software Role",
    )

    result["skills_present"] = _ensure_list(
        result.get(
            "skills_present",
            [],
        )
    )

    result["skills_missing"] = _ensure_list(
        result.get(
            "skills_missing",
            [],
        )
    )

    skill_strength = result.get(
        "skill_strength",
        {},
    )

    if not isinstance(
        skill_strength,
        dict,
    ):
        skill_strength = {}

    skill_strength.setdefault(
        "strong",
        [],
    )

    skill_strength.setdefault(
        "moderate",
        [],
    )

    skill_strength.setdefault(
        "weak_or_buzzword",
        [],
    )

    skill_strength["strong"] = _ensure_list(
        skill_strength["strong"]
    )

    skill_strength["moderate"] = _ensure_list(
        skill_strength["moderate"]
    )

    skill_strength["weak_or_buzzword"] = _ensure_list(
        skill_strength["weak_or_buzzword"]
    )

    result["skill_strength"] = skill_strength

    result["recommended_skills_to_learn"] = _ensure_list(
        result.get(
            "recommended_skills_to_learn",
            [],
        )
    )

    return result


# ==============================================================
# COMBINED RESUME + SKILL ANALYSIS
# ==============================================================

def analyze_resume_and_skills(
    structured: dict,
    resume_text: str,
    candidate_type: str,
    llm: Optional[LLMClient] = None,
) -> dict:

    llm = llm or LLMClient()

    prompt = COMBINED_RESUME_ANALYSIS_PROMPT.format(
        candidate_type=candidate_type,
        structured_data=json.dumps(
            structured,
            indent=2,
        ),
        resume_text=resume_text[:6000],
    )

    result = llm.chat_json(
        RECRUITER_SYSTEM_PROMPT,
        prompt,
        temperature=0.1,
        max_tokens=3000,
    )

    # ----------------------------------------------------------
    # Resume score
    # ----------------------------------------------------------

    resume_score = result.get(
        "resume_score",
        {},
    )

    if not isinstance(
        resume_score,
        dict,
    ):
        resume_score = {}

    resume_score["overall_score"] = _clamp_score(
        resume_score.get(
            "overall_score",
            0,
        )
    )

    category_scores = resume_score.get(
        "category_scores",
        {},
    )

    if not isinstance(
        category_scores,
        dict,
    ):
        category_scores = {}

    score_categories = (
        "ats_compatibility",
        "formatting",
        "technical_skills",
        "projects",
        "experience",
        "education",
        "achievements",
        "grammar",
    )

    for key in score_categories:

        category_scores[key] = _clamp_score(
            category_scores.get(
                key,
                0,
            )
        )

    resume_score["category_scores"] = category_scores

    resume_score["strengths"] = _ensure_list(
        resume_score.get(
            "strengths",
            [],
        )
    )

    resume_score["weaknesses"] = _ensure_list(
        resume_score.get(
            "weaknesses",
            [],
        )
    )

    # ----------------------------------------------------------
    # Skill analysis
    # ----------------------------------------------------------

    skill_analysis = result.get(
        "skill_analysis",
        {},
    )

    if not isinstance(
        skill_analysis,
        dict,
    ):
        skill_analysis = {}

    skill_analysis.setdefault(
        "inferred_target_role",
        "General Software Role",
    )

    skill_analysis["skills_present"] = _ensure_list(
        skill_analysis.get(
            "skills_present",
            [],
        )
    )

    skill_analysis["skills_missing"] = _ensure_list(
        skill_analysis.get(
            "skills_missing",
            [],
        )
    )

    skill_strength = skill_analysis.get(
        "skill_strength",
        {},
    )

    if not isinstance(
        skill_strength,
        dict,
    ):
        skill_strength = {}

    skill_strength.setdefault(
        "strong",
        [],
    )

    skill_strength.setdefault(
        "moderate",
        [],
    )

    skill_strength.setdefault(
        "weak_or_buzzword",
        [],
    )

    skill_strength["strong"] = _ensure_list(
        skill_strength["strong"]
    )

    skill_strength["moderate"] = _ensure_list(
        skill_strength["moderate"]
    )

    skill_strength["weak_or_buzzword"] = _ensure_list(
        skill_strength["weak_or_buzzword"]
    )

    skill_analysis["skill_strength"] = skill_strength

    skill_analysis["recommended_skills_to_learn"] = _ensure_list(
        skill_analysis.get(
            "recommended_skills_to_learn",
            [],
        )
    )

    return {
        "resume_score": resume_score,
        "skill_analysis": skill_analysis,
    }


# ==============================================================
# STEP 7: PROJECT EVALUATION
# ==============================================================

def evaluate_projects(
    structured: dict,
    candidate_type: str,
    llm: Optional[LLMClient] = None,
) -> dict:

    llm = llm or LLMClient()

    projects = structured.get(
        "projects",
        [],
    )

    if not projects:
        return {
            "projects": []
        }

    prompt = PROJECT_EVALUATION_PROMPT.format(
        candidate_type=candidate_type,
        projects_json=json.dumps(
            projects,
            indent=2,
        ),
    )

    result = llm.chat_json(
        RECRUITER_SYSTEM_PROMPT,
        prompt,
        temperature=0.2,
        max_tokens=2500,
    )

    result_projects = result.get(
        "projects",
        [],
    )

    if not isinstance(
        result_projects,
        list,
    ):
        result_projects = []

    # Project scores are 0-10.
    for project in result_projects:

        if isinstance(
            project,
            dict,
        ):

            project["score"] = _clamp_score(
                project.get(
                    "score",
                    0,
                ),
                low=0,
                high=10,
            )

            project.setdefault(
                "name",
                "",
            )

            project.setdefault(
                "problem_statement_clarity",
                "",
            )

            project.setdefault(
                "tech_stack_assessment",
                "",
            )

            project.setdefault(
                "complexity",
                "Medium",
            )

            project.setdefault(
                "innovation",
                "",
            )

            project.setdefault(
                "real_world_usage",
                "",
            )

            project.setdefault(
                "has_measurable_outcomes",
                False,
            )

            project.setdefault(
                "improved_description",
                "",
            )

    return {
        "projects": result_projects
    }


# ==============================================================
# STEP 8: EXPERIENCE EVALUATION
# ==============================================================

def evaluate_experience(
    structured: dict,
    candidate_type: str,
    llm: Optional[LLMClient] = None,
) -> dict:

    llm = llm or LLMClient()

    experience = structured.get(
        "experience",
        [],
    )

    if not experience:

        return {
            "entries": [],
            "overall_notes": (
                "No work or internship experience "
                "was found on the resume. "
                "This should not be heavily penalized "
                "for a Student or Fresher profile. "
                "Focus evaluation on projects, "
                "skills, and education."
            ),
        }

    prompt = EXPERIENCE_EVALUATION_PROMPT.format(
        candidate_type=candidate_type,
        experience_json=json.dumps(
            experience,
            indent=2,
        ),
    )

    result = llm.chat_json(
        RECRUITER_SYSTEM_PROMPT,
        prompt,
        temperature=0.2,
        max_tokens=2000,
    )

    result["entries"] = _ensure_list(
        result.get(
            "entries",
            [],
        )
    )

    result.setdefault(
        "overall_notes",
        "",
    )

    return result


# ==============================================================
# STEP 9: JOB DESCRIPTION MATCHING
# ==============================================================

def match_job_description(
    structured: dict,
    job_description: str,
    llm: Optional[LLMClient] = None,
) -> dict:

    llm = llm or LLMClient()

    if not job_description or not job_description.strip():

        return {
            "match_percentage": 0,
            "matching_skills": [],
            "missing_skills": [],
            "missing_keywords": [],
            "improvement_suggestions": [
                "Provide a job description to calculate a match."
            ],
        }

    prompt = JD_MATCH_PROMPT.format(
        structured_data=json.dumps(
            structured,
            indent=2,
        ),
        job_description=job_description[:6000],
    )

    result = llm.chat_json(
        RECRUITER_SYSTEM_PROMPT,
        prompt,
        temperature=0.1,
        max_tokens=1800,
    )

    result["match_percentage"] = _clamp_score(
        result.get(
            "match_percentage",
            0,
        )
    )

    result["matching_skills"] = _ensure_list(
        result.get(
            "matching_skills",
            [],
        )
    )

    result["missing_skills"] = _ensure_list(
        result.get(
            "missing_skills",
            [],
        )
    )

    result["missing_keywords"] = _ensure_list(
        result.get(
            "missing_keywords",
            [],
        )
    )

    result["improvement_suggestions"] = _ensure_list(
        result.get(
            "improvement_suggestions",
            [],
        )
    )

    return result