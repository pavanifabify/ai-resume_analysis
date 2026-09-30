"""
app.py
------
Main Streamlit application.

Optimized pipeline:

Upload
  -> Extract
  -> Parse                 [1 LLM call]
  -> Resume + Skill Analysis [1 LLM call]
  -> Local candidate classification
  -> Local project/experience display
  -> Local career roadmap
  -> Report
  -> Save to DB

Optional LLM calls:
  -> Job Description Matching
  -> RAG Q&A

The goal is to avoid repeatedly sending the same resume information
to the LLM for separate project, experience, classification and roadmap calls.
"""

import json
import traceback

import streamlit as st

from src.config import Config
from src import database as db
from src.resume_parser import extract_text, parse_resume_structured
from src.analyzer import (
    analyze_resume_and_skills,
    match_job_description,
)
from src.rag_pipeline import build_index, rag_answer
from src.llm_client import LLMClient, LLMError


st.set_page_config(
    page_title="AI Resume Analyzer",
    page_icon="📄",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def setup():
    db.init_db()

    try:
        build_index()
        kb_ready = True
        kb_error = None
    except Exception as e:
        kb_ready = False
        kb_error = str(e)

    return kb_ready, kb_error


def get_llm():
    if "llm_client" not in st.session_state:
        st.session_state.llm_client = LLMClient()

    return st.session_state.llm_client


# ---------------------------------------------------------------------------
# Local helper functions
# ---------------------------------------------------------------------------

def score_color(score: int) -> str:
    if score >= 80:
        return "🟢"

    if score >= 60:
        return "🟡"

    return "🔴"


def infer_candidate_type(structured: dict, target_role: str) -> dict:
    """
    Local candidate classification.

    This replaces the separate classify_candidate() LLM call.
    """

    experience = structured.get("experience", [])

    if experience:
        candidate_type = "Experienced Candidate"
        reason = "The resume contains professional or internship experience."
        confidence = "High"
    else:
        candidate_type = "Fresher"
        reason = "No professional experience was detected in the structured resume."
        confidence = "High"

    if target_role:
        reason += f" Target role inferred as {target_role}."

    return {
        "candidate_type": candidate_type,
        "reason": reason,
        "confidence": confidence,
    }


def generate_local_roadmap(skills: dict) -> dict:
    """
    Generate a simple roadmap locally.

    This replaces the separate generate_career_roadmap() LLM call.
    """

    missing_skills = skills.get("skills_missing", [])
    target_role = skills.get("inferred_target_role", "target role")

    short_term = [
        f"Strengthen the core skills required for {target_role}.",
        "Improve resume bullets using measurable results and action verbs.",
        "Build or improve one role-relevant project.",
        "Practice technical interview fundamentals.",
    ]

    if missing_skills:
        short_term.append(
            "Prioritize these missing skills: "
            + ", ".join(missing_skills[:5])
        )

    mid_term = [
        f"Build 1–2 stronger projects aligned with {target_role}.",
        "Practice coding and technical interview questions regularly.",
        "Apply to relevant internships and entry-level positions.",
        "Improve ATS keyword alignment for each job description.",
    ]

    long_term = [
        f"Develop deeper expertise in {target_role}.",
        "Build a portfolio demonstrating practical experience.",
        "Prepare for role-specific interviews and system/design discussions.",
        "Continuously update the resume with measurable achievements.",
    ]

    resources = [
        "LeetCode — coding interview practice",
        "HackerRank — SQL and programming practice",
        "GitHub — project portfolio",
        "Official documentation for your target technologies",
    ]

    return {
        "short_term_0_3_months": short_term,
        "mid_term_3_9_months": mid_term,
        "long_term_9_18_months": long_term,
        "recommended_resources": resources,
    }


def build_project_evaluation_locally(structured: dict) -> dict:
    """
    Display project information without making another LLM call.

    We intentionally do not fabricate project scores.
    """

    projects = structured.get("projects", [])

    result = []

    for project in projects:
        name = project.get("name", "Unnamed Project")

        description = (
            project.get("description")
            or project.get("problem_statement")
            or project.get("details")
            or "Project description available in resume."
        )

        tech_stack = (
            project.get("technologies")
            or project.get("tech_stack")
            or project.get("tools")
            or []
        )

        if isinstance(tech_stack, list):
            tech_stack_text = ", ".join(map(str, tech_stack))
        else:
            tech_stack_text = str(tech_stack)

        result.append(
            {
                "name": name,
                "score": "N/A",
                "complexity": "Not separately evaluated to avoid an additional LLM call.",
                "problem_statement_clarity": description,
                "tech_stack_assessment": tech_stack_text or "Not specified",
                "innovation": "Not separately evaluated.",
                "real_world_usage": "Not separately evaluated.",
                "has_measurable_outcomes": False,
                "improved_description": description,
            }
        )

    return {
        "projects": result,
    }


def build_experience_evaluation_locally(structured: dict) -> dict:
    """
    Display experience without making another LLM call.
    """

    entries = []

    for experience in structured.get("experience", []):
        role = (
            experience.get("role")
            or experience.get("title")
            or "Unknown Role"
        )

        company = (
            experience.get("company")
            or experience.get("organization")
            or "Unknown Company"
        )

        bullets = (
            experience.get("responsibilities")
            or experience.get("description")
            or experience.get("bullets")
            or []
        )

        if isinstance(bullets, str):
            bullets = [bullets]

        entries.append(
            {
                "role": role,
                "company": company,
                "date_consistency_issue": None,
                "weak_description_bullets": [],
                "suggested_rewrite": bullets,
            }
        )

    return {
        "overall_notes": (
            "Experience information was extracted from the resume. "
            "Detailed LLM-based experience evaluation was skipped to reduce API calls."
        ),
        "entries": entries,
    }


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------

def render_score_block(title: str, scores: dict):
    st.subheader(title)

    if not scores:
        st.info("No category scores available.")
        return

    cols = st.columns(len(scores))

    for col, (key, val) in zip(cols, scores.items()):
        with col:
            st.metric(
                key.replace("_", " ").title(),
                f"{val}/100",
            )


def render_report(report: dict):

    profile = report["candidate_profile"]
    scoring = report["scoring"]
    skills = report["skill_analysis"]
    projects = report["project_evaluation"]
    experience = report["experience_evaluation"]

    # ------------------------------------------------------------------
    # Candidate Profile
    # ------------------------------------------------------------------

    st.header("👤 Candidate Profile")

    c1, c2 = st.columns([2, 1])

    with c1:
        st.markdown(
            f"**Candidate Type:** {profile['candidate_type']}"
        )

        st.markdown(
            f"**Reason:** {profile['reason']}"
        )

    with c2:
        st.markdown(
            f"**Confidence:** {profile['confidence']}"
        )

    st.divider()

    # ------------------------------------------------------------------
    # Resume Score
    # ------------------------------------------------------------------

    st.header("📊 Resume Score")

    overall = scoring.get("overall_score", 0)

    st.markdown(
        f"## {score_color(overall)} Overall Score: {overall}/100"
    )

    render_score_block(
        "Category Breakdown",
        scoring.get("category_scores", {}),
    )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("✅ Strengths")

        for strength in scoring.get("strengths", []):
            st.markdown(f"- {strength}")

    with col2:
        st.subheader("⚠️ Weaknesses")

        for weakness in scoring.get("weaknesses", []):
            st.markdown(f"- {weakness}")

    st.divider()

    # ------------------------------------------------------------------
    # Skills
    # ------------------------------------------------------------------

    st.header("🛠️ Skill Analysis")

    target_role = skills.get(
        "inferred_target_role",
        "Not determined",
    )

    st.markdown(
        f"**Inferred Target Role:** {target_role}"
    )

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Present")

        st.write(
            ", ".join(
                skills.get("skills_present", [])
            )
            or "None detected"
        )

        st.subheader("Strong")

        strong = skills.get(
            "skill_strength", {}
        ).get("strong", [])

        st.write(
            ", ".join(strong)
            or "None"
        )

    with col2:

        st.subheader("Missing")

        st.write(
            ", ".join(
                skills.get("skills_missing", [])
            )
            or "None"
        )

        st.subheader("Weak / Buzzword only")

        weak = skills.get(
            "skill_strength", {}
        ).get("weak_or_buzzword", [])

        st.write(
            ", ".join(weak)
            or "None"
        )

    st.subheader("📚 Recommended Skills to Learn")

    for skill in skills.get(
        "recommended_skills_to_learn",
        [],
    ):
        st.markdown(f"- {skill}")

    st.divider()

    # ------------------------------------------------------------------
    # Projects
    # ------------------------------------------------------------------

    st.header("💼 Project Evaluation")

    if not projects.get("projects"):
        st.info("No projects found on this resume.")

    for project in projects.get("projects", []):

        with st.expander(
            f"{project['name']} — {project['score']}"
        ):

            st.markdown(
                f"**Description:** "
                f"{project['problem_statement_clarity']}"
            )

            st.markdown(
                f"**Tech stack:** "
                f"{project['tech_stack_assessment']}"
            )

            st.info(
                "Detailed project scoring was skipped in the optimized "
                "pipeline to avoid another LLM call."
            )

    st.divider()

    # ------------------------------------------------------------------
    # Experience
    # ------------------------------------------------------------------

    st.header("🏢 Experience / Internship")

    st.write(
        experience.get(
            "overall_notes",
            "",
        )
    )

    for entry in experience.get("entries", []):

        with st.expander(
            f"{entry['role']} @ {entry['company']}"
        ):

            for bullet in entry.get(
                "suggested_rewrite",
                [],
            ):
                st.markdown(f"- {bullet}")


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:

    st.title("📄 AI Resume Analyzer")

    st.caption("Powered by LLM + RAG")

    kb_ready, kb_error = setup()

    if not kb_ready:
        st.error(
            f"Knowledge base failed to load: {kb_error}"
        )

    problems = Config.validate()

    if problems:

        st.error(
            "Configuration issues:\n"
            + "\n".join(
                f"- {problem}"
                for problem in problems
            )
        )

        st.info(
            "Add your API key to a `.env` file. "
            "See README.md for setup instructions."
        )

    st.markdown("---")

    st.markdown(
        f"**LLM Provider:** `{Config.LLM_PROVIDER}`"
    )

    st.markdown(
        "**Formats supported:** PDF, DOCX"
    )

    st.markdown("---")

    with st.expander("📜 Past Analyses"):

        history = db.get_resume_history()

        if not history:
            st.caption(
                "No resumes analyzed yet."
            )

        for item in history[:10]:

            st.caption(
                f"{item['resume_filename']} — "
                f"{item['upload_date'][:10]}"
            )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

st.title("AI Resume Analyzer")

st.write(
    "Upload your resume to get a recruiter-style analysis: "
    "resume score, ATS compatibility, skill gaps, project feedback, "
    "and a personalized career roadmap."
)

uploaded_file = st.file_uploader(
    "Upload your resume (PDF or DOCX)",
    type=["pdf", "docx"],
)


if uploaded_file is not None and Config.validate():
    st.stop()


if uploaded_file is not None:

    file_bytes = uploaded_file.read()

    if st.button(
        "🔍 Analyze Resume",
        type="primary",
    ):

        try:

            llm = get_llm()

            with st.status(
                "Running optimized analysis...",
                expanded=True,
            ):

                # ------------------------------------------------------
                # CALL #1
                # ------------------------------------------------------

                st.write(
                    "📄 Extracting text..."
                )

                resume_text = extract_text(
                    file_bytes,
                    uploaded_file.name,
                )

                st.write(
                    "🧩 Parsing resume into structured data..."
                )

                structured = parse_resume_structured(
                    resume_text,
                    llm,
                )

                # ------------------------------------------------------
                # CALL #2
                # ------------------------------------------------------

                st.write(
                    "📊 Running combined resume + skill analysis..."
                )

                combined_analysis = analyze_resume_and_skills(
                    structured,
                    resume_text,
                    "candidate",
                    llm,
                )

                scoring = combined_analysis.get(
                    "resume_score",
                    {},
                )

                skills = combined_analysis.get(
                    "skill_analysis",
                    {},
                )

                # ------------------------------------------------------
                # NO EXTRA LLM CALLS
                # ------------------------------------------------------

                st.write(
                    "🏷️ Determining candidate profile locally..."
                )

                profile = infer_candidate_type(
                    structured,
                    skills.get(
                        "inferred_target_role",
                        "",
                    ),
                )

                st.write(
                    "💼 Preparing project information..."
                )

                projects = build_project_evaluation_locally(
                    structured
                )

                st.write(
                    "🏢 Preparing experience information..."
                )

                experience = build_experience_evaluation_locally(
                    structured
                )

                st.write(
                    "🗺️ Building career roadmap..."
                )

                roadmap = generate_local_roadmap(
                    skills
                )

                st.write(
                    "💾 Preparing report..."
                )

                status = st.session_state

            report = {
                "candidate_profile": profile,
                "structured_data": structured,
                "scoring": scoring,
                "skill_analysis": skills,
                "project_evaluation": projects,
                "experience_evaluation": experience,
                "career_roadmap": roadmap,
            }

            # ----------------------------------------------------------
            # Save to DB
            # ----------------------------------------------------------

            personal = structured.get(
                "personal_info",
                {},
            )

            user_id = db.get_or_create_user(
                personal.get("name"),
                personal.get("email"),
            )

            resume_history_id = db.save_resume_history(
                user_id,
                uploaded_file.name,
                resume_text,
            )

            db.save_analysis_report(
                resume_history_id,
                profile["candidate_type"],
                scoring.get("overall_score", 0),
                report,
            )

            st.session_state.report = report
            st.session_state.resume_history_id = resume_history_id
            st.session_state.structured = structured

            st.success(
                "Analysis complete using 2 LLM calls."
            )

        except LLMError as e:

            st.error(
                f"LLM error: {e}"
            )

        except ValueError as e:

            st.error(
                str(e)
            )

        except Exception as e:

            st.error(
                f"Unexpected error: {e}"
            )

            with st.expander("Details"):

                st.code(
                    traceback.format_exc()
                )


# ---------------------------------------------------------------------------
# Render report + downstream features
# ---------------------------------------------------------------------------

if "report" in st.session_state:

    st.divider()

    render_report(
        st.session_state.report
    )

    # ------------------------------------------------------------------
    # Career Roadmap
    # ------------------------------------------------------------------

    st.divider()

    st.header("🗺️ Career Roadmap")

    roadmap = st.session_state.report[
        "career_roadmap"
    ]

    c1, c2, c3 = st.columns(3)

    with c1:

        st.subheader("0-3 months")

        for item in roadmap.get(
            "short_term_0_3_months",
            [],
        ):
            st.markdown(f"- {item}")

    with c2:

        st.subheader("3-9 months")

        for item in roadmap.get(
            "mid_term_3_9_months",
            [],
        ):
            st.markdown(f"- {item}")

    with c3:

        st.subheader("9-18 months")

        for item in roadmap.get(
            "long_term_9_18_months",
            [],
        ):
            st.markdown(f"- {item}")

    st.subheader(
        "Recommended Resources"
    )

    for resource in roadmap.get(
        "recommended_resources",
        [],
    ):
        st.markdown(f"- {resource}")

    # ------------------------------------------------------------------
    # Job Description Matching
    # ------------------------------------------------------------------

    st.divider()

    st.header(
        "🎯 Job Description Matching"
    )

    jd_text = st.text_area(
        "Paste a job description to compare against your resume",
        height=200,
    )

    if st.button(
        "Compare with Job Description"
    ):

        if not jd_text.strip():

            st.warning(
                "Please paste a job description first."
            )

        else:

            with st.spinner(
                "Comparing resume against job description..."
            ):

                try:

                    match = match_job_description(
                        st.session_state.structured,
                        jd_text,
                        get_llm(),
                    )

                    st.session_state.jd_match = match

                    db.save_job_description(
                        st.session_state.resume_history_id,
                        jd_text,
                        match,
                    )

                except Exception as e:

                    st.error(
                        f"Matching failed: {e}"
                    )

    if "jd_match" in st.session_state:

        match = st.session_state.jd_match

        st.markdown(
            f"### Match: "
            f"{score_color(match['match_percentage'])} "
            f"{match['match_percentage']}%"
        )

        col1, col2 = st.columns(2)

        with col1:

            st.subheader(
                "Matching Skills"
            )

            st.write(
                ", ".join(
                    match.get(
                        "matching_skills",
                        [],
                    )
                )
                or "None"
            )

        with col2:

            st.subheader(
                "Missing Skills"
            )

            st.write(
                ", ".join(
                    match.get(
                        "missing_skills",
                        [],
                    )
                )
                or "None"
            )

        st.subheader(
            "Missing Keywords (ATS)"
        )

        st.write(
            ", ".join(
                match.get(
                    "missing_keywords",
                    [],
                )
            )
            or "None"
        )

        st.subheader(
            "Improvement Suggestions"
        )

        for suggestion in match.get(
            "improvement_suggestions",
            [],
        ):
            st.markdown(
                f"- {suggestion}"
            )

    # ------------------------------------------------------------------
    # RAG Q&A
    # ------------------------------------------------------------------

    st.divider()

    st.header(
        "💬 Ask a Career/Resume Question"
    )

    question = st.text_input(
        "e.g. 'How do I make my ATS score higher?' "
        "or 'What certifications should I pursue?'"
    )

    if st.button("Ask"):

        if not question.strip():

            st.warning(
                "Please enter a question."
            )

        else:

            with st.spinner(
                "Thinking..."
            ):

                try:

                    candidate_summary = json.dumps(
                        {
                            "candidate_profile": st.session_state.report[
                                "candidate_profile"
                            ],
                            "target_role": st.session_state.report[
                                "skill_analysis"
                            ].get(
                                "inferred_target_role"
                            ),
                            "skills": st.session_state.report[
                                "skill_analysis"
                            ],
                        }
                    )

                    answer = rag_answer(
                        question,
                        candidate_summary,
                        get_llm(),
                    )

                    st.markdown(
                        answer["answer"]
                    )

                    if answer.get("sources"):

                        st.caption(
                            "Sources: "
                            + ", ".join(
                                set(answer["sources"])
                            )
                        )

                except Exception as e:

                    st.error(
                        f"Question answering failed: {e}"
                    )

    # ------------------------------------------------------------------
    # Download report
    # ------------------------------------------------------------------

    st.divider()

    st.download_button(
        "⬇️ Download Full Report (JSON)",
        data=json.dumps(
            st.session_state.report,
            indent=2,
        ),
        file_name="resume_analysis_report.json",
        mime="application/json",
    )