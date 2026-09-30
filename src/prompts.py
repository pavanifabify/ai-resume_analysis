"""
prompts.py
----------
Centralized LLM prompts for the AI Resume Analyzer.

All prompts are designed to:
- Return predictable JSON.
- Avoid fabricated resume information.
- Calibrate feedback to the candidate's career stage.
- Produce evidence-based recruiter feedback.
- Keep output compatible with the schemas expected by analyzer.py,
  candidate_classifier.py, resume_parser.py, and rag_pipeline.py.
"""


# ============================================================================
# RECRUITER PERSONA
# ============================================================================

RECRUITER_PERSONA = (
    "You are a senior technical recruiter and career coach with extensive "
    "experience evaluating software engineering, QA, testing, data, cloud, "
    "and other technical candidates. "
    "You are direct, specific, practical, and evidence-based. "
    "You never invent facts that are not present in the candidate's resume. "
    "You clearly distinguish between evidence, inference, and recommendation. "
    "You calibrate expectations to the candidate's actual career stage. "
    "Do not penalize students or freshers simply because they lack professional "
    "experience. "
    "When recommending improvements, prioritize changes that are realistically "
    "useful for getting interviews."
)


# ============================================================================
# RESUME PARSING
# ============================================================================

RESUME_PARSE_PROMPT = """
Extract structured information from the resume text below.

Your task is EXTRACTION, not evaluation.

IMPORTANT RULES:
1. Extract only information explicitly present in the resume.
2. Never invent a company, degree, skill, project, certification, date,
   achievement, score, or link.
3. Do not infer a technology merely because it is commonly associated with
   another technology.
4. Preserve the candidate's actual wording where practical.
5. If information is missing, use an empty string, null, or empty list.
6. Do not create information to fill missing fields.
7. Keep separate projects, education entries, experience entries, and
   certifications separate.
8. Contact information should be copied exactly when present.
9. Do not treat generic words such as "development", "programming", or
   "computer science" as specific technologies.
10. Return ONLY JSON.

Resume text:
---
{resume_text}
---

Return JSON with EXACTLY this structure:

{{
  "personal_info": {{
    "name": "",
    "email": "",
    "phone": "",
    "linkedin": "",
    "github": "",
    "portfolio": ""
  }},

  "education": [
    {{
      "degree": "",
      "college": "",
      "graduation_year": "",
      "score": ""
    }}
  ],

  "skills": {{
    "programming_languages": [],
    "frameworks": [],
    "databases": [],
    "tools": [],
    "cloud": [],
    "testing": []
  }},

  "experience": [
    {{
      "company": "",
      "role": "",
      "start_date": "",
      "end_date": "",
      "duration": "",
      "responsibilities": []
    }}
  ],

  "projects": [
    {{
      "name": "",
      "technologies": [],
      "description": "",
      "duration": "",
      "link": ""
    }}
  ],

  "certifications": [
    {{
      "name": "",
      "provider": "",
      "date": ""
    }}
  ],

  "achievements": []
}}
"""


# ============================================================================
# CANDIDATE CLASSIFICATION
# ============================================================================

CANDIDATE_CLASSIFICATION_PROMPT = """
Classify the candidate into EXACTLY ONE of the following categories.

Allowed categories:

1. Student
   - Currently enrolled.
   - Usually no full-time professional experience.
   - May have internships or academic projects.

2. Fresher
   - Recently graduated, normally within approximately one year.
   - No meaningful full-time professional experience.

3. Internship Candidate
   - Currently seeking internships or clearly positioned for internship roles.
   - Usually still studying or recently graduated.
   - Internship experience may already exist.

4. Entry Level Professional
   - Has approximately 0-2 years of full-time professional experience.

5. Experienced Professional
   - Has approximately 2+ years of full-time professional experience.

6. Career Switcher
   - Has professional experience in one field but is now clearly targeting
     a substantially different field.

IMPORTANT:
- Use the structured resume data as evidence.
- Use the rule-based signals as strong hints.
- Do not classify someone as a Career Switcher merely because they have
  learned a new technology.
- Do not classify internships as full-time professional experience.
- Do not assume a graduation year if it is missing.
- If evidence is ambiguous, choose the category best supported by the
  available evidence and use lower confidence.
- The reason must cite specific evidence.
- Return ONLY JSON.

Structured resume data:
---
{structured_data}
---

Rule-based signals:
---
{rule_based_hints}
---

Return EXACTLY:

{{
  "candidate_type": "Student",
  "confidence": "High",
  "reason": "2-3 sentences explaining the classification using evidence."
}}

candidate_type MUST be exactly one of:

"Student"
"Fresher"
"Internship Candidate"
"Entry Level Professional"
"Experienced Professional"
"Career Switcher"

confidence MUST be exactly one of:

"High"
"Medium"
"Low"
"""


# ============================================================================
# RESUME SCORING
# ============================================================================

SCORING_PROMPT = """
Evaluate the following resume for a candidate classified as:

{candidate_type}

IMPORTANT:
The score must reflect the candidate's career stage.

For example:
- A Student/Fresher should not be heavily penalized for having no full-time
  professional experience.
- For students and freshers, projects, internships, education, skills,
  certifications, and demonstrated technical ability matter more.
- Experienced professionals should be evaluated more heavily on professional
  impact and measurable achievements.

Structured resume data:
---
{structured_data}
---

Raw resume text:
---
{resume_text}
---

Evaluate these categories from 0-100:

1. ATS compatibility
2. Formatting
3. Technical skills
4. Projects
5. Experience
6. Education
7. Achievements
8. Grammar

Scoring guidance:

90-100 = Excellent
80-89  = Strong
70-79  = Good
60-69  = Average
40-59  = Weak
0-39   = Poor

IMPORTANT EVIDENCE RULES:
- Base feedback only on the resume.
- Do not invent achievements.
- Do not invent metrics.
- Do not claim ATS compatibility with certainty because actual ATS software
  has not been run.
- Evaluate ATS compatibility based on observable resume structure, keywords,
  headings, formatting, and readability.
- Weaknesses must identify an actual improvement opportunity.
- Strengths must identify an actual positive feature.
- Give 3-6 strengths.
- Give 3-6 weaknesses.

Return ONLY JSON.

Return EXACTLY:

{{
  "overall_score": 0,

  "category_scores": {{
    "ats_compatibility": 0,
    "formatting": 0,
    "technical_skills": 0,
    "projects": 0,
    "experience": 0,
    "education": 0,
    "achievements": 0,
    "grammar": 0
  }},

  "strengths": [
    "...",
    "..."
  ],

  "weaknesses": [
    "...",
    "..."
  ]
}}
"""


# ============================================================================
# SKILL ANALYSIS
# ============================================================================

SKILL_ANALYSIS_PROMPT = """
Analyze the candidate's skills and determine their likely target technical role.

Candidate type:
{candidate_type}

Extracted skills:
---
{skills_json}
---

Projects and experience:
---
{projects_and_experience}
---

IMPORTANT:
1. Do not assume a target role without evidence.
2. Infer the role from the combination of skills, projects, education,
   and experience.
3. If evidence is insufficient, use a broad role such as:
   "General Software Role"
4. A skill should be considered STRONG only when there is meaningful evidence
   of usage through projects, work, internships, or repeated demonstrations.
5. A skill mentioned only once without supporting evidence should generally
   be MODERATE.
6. A skill listed without any evidence should generally be considered
   WEAK_OR_BUZZWORD.
7. Missing skills should be relevant to the inferred target role.
8. Do not recommend a huge list of technologies.
9. Prioritize 3-6 skills that would have the highest practical impact.

Return ONLY JSON.

Return EXACTLY:

{{
  "inferred_target_role": "",

  "skills_present": [],

  "skills_missing": [],

  "skill_strength": {{
    "strong": [],
    "moderate": [],
    "weak_or_buzzword": []
  }},

  "recommended_skills_to_learn": []
}}
"""


# ============================================================================
# COMBINED RESUME + SKILL ANALYSIS
# ============================================================================

COMBINED_RESUME_ANALYSIS_PROMPT = """
You are evaluating a resume for:

Candidate type:
{candidate_type}

Perform TWO analyses:

A. Resume quality scoring
B. Skill analysis

==================================================
STRUCTURED RESUME DATA
==================================================

{structured_data}

==================================================
RAW RESUME TEXT
==================================================

{resume_text}

==================================================
A. RESUME SCORING
==================================================

Score each category from 0-100:

- ATS compatibility
- Formatting
- Technical skills
- Projects
- Experience
- Education
- Achievements
- Grammar

Give an overall score from 0-100.

IMPORTANT:
- Calibrate the score to the candidate type.
- Do not heavily penalize a Student/Fresher for lacking full-time experience.
- Do not invent achievements or metrics.
- Do not claim that a resume is guaranteed to pass an ATS.
- Give concrete evidence-based strengths.
- Give concrete evidence-based weaknesses.

==================================================
B. SKILL ANALYSIS
==================================================

Determine:

- Most likely target role.
- Skills currently present.
- Skills missing for that target role.
- Strong skills.
- Moderate skills.
- Weak/buzzword skills.
- 3-6 highest-priority skills to learn.

IMPORTANT:
Do not recommend skills simply because they are popular.
Recommendations must be relevant to the candidate's inferred target role.

==================================================
OUTPUT
==================================================

Return ONLY valid JSON.

Return EXACTLY:

{{
  "resume_score": {{
    "overall_score": 0,

    "category_scores": {{
      "ats_compatibility": 0,
      "formatting": 0,
      "technical_skills": 0,
      "projects": 0,
      "experience": 0,
      "education": 0,
      "achievements": 0,
      "grammar": 0
    }},

    "strengths": [],

    "weaknesses": []
  }},

  "skill_analysis": {{
    "inferred_target_role": "",
    "skills_present": [],
    "skills_missing": [],

    "skill_strength": {{
      "strong": [],
      "moderate": [],
      "weak_or_buzzword": []
    }},

    "recommended_skills_to_learn": []
  }}
}}
"""


# ============================================================================
# PROJECT EVALUATION
# ============================================================================

PROJECT_EVALUATION_PROMPT = """
Evaluate EACH project independently.

Candidate type:
{candidate_type}

Projects:
---
{projects_json}
---

For every project evaluate:

1. Problem statement clarity
2. Technology stack appropriateness
3. Complexity
4. Innovation
5. Real-world usage potential
6. Description quality
7. Whether measurable outcomes are actually stated

IMPORTANT:
- Do not invent metrics.
- Do not create fake users, performance improvements, accuracy percentages,
  revenue, or other outcomes.
- If measurable impact is absent, say so.
- A student project can still receive a strong score if it demonstrates
  meaningful technical depth.
- Do not judge academic projects as if they were enterprise production systems.
- Score each project from 0-10.

Return ONLY JSON.

Return EXACTLY:

{{
  "projects": [
    {{
      "name": "",
      "score": 0,
      "problem_statement_clarity": "",
      "tech_stack_assessment": "",
      "complexity": "Low",
      "innovation": "",
      "real_world_usage": "",
      "has_measurable_outcomes": false,
      "improved_description": ""
    }}
  ]
}}

complexity MUST be one of:

"Low"
"Medium"
"High"
"""


# ============================================================================
# EXPERIENCE EVALUATION
# ============================================================================

EXPERIENCE_EVALUATION_PROMPT = """
Evaluate the following work and internship experience.

Candidate type:
{candidate_type}

Experience:
---
{experience_json}
---

Evaluate:

1. Date consistency
2. Clarity of responsibilities
3. Presence of measurable achievements
4. Strength of descriptions
5. Use of action verbs
6. Evidence of impact

IMPORTANT:
- Do not invent metrics.
- Do not invent achievements.
- If metrics are missing, explicitly state that they are missing.
- Internship experience should be evaluated appropriately for the candidate's
  career stage.
- Suggested rewrites may include placeholders such as "[X%]" or "[N users]"
  when a real metric would strengthen the bullet, but do NOT present invented
  numbers as facts.

Return ONLY JSON.

Return EXACTLY:

{{
  "entries": [
    {{
      "company": "",
      "role": "",
      "date_consistency_issue": null,
      "missing_achievements": true,
      "weak_description_bullets": [],
      "lacks_measurable_impact": true,
      "suggested_rewrite": []
    }}
  ],

  "overall_notes": ""
}}
"""


# ============================================================================
# JOB DESCRIPTION MATCHING
# ============================================================================

JD_MATCH_PROMPT = """
Compare the candidate's resume against the job description.

Resume structured data:
---
{structured_data}
---

Job description:
---
{job_description}
---

Determine:

1. Match percentage
2. Skills explicitly supported by the resume
3. Skills required by the JD but not evidenced in the resume
4. Important JD keywords missing from the resume
5. Specific improvements that could increase the candidate's fit

IMPORTANT:
- Do not inflate the match percentage.
- Do not treat a skill as present merely because it is similar to another skill.
- Distinguish exact evidence from assumptions.
- If the JD requires a skill that is not present in the resume, list it as missing.
- Missing keywords should be useful for ATS optimization.
- Do not tell the candidate to add a keyword unless they actually have that
  skill or experience.
- Give 3-6 actionable suggestions.
- Return ONLY JSON.

Return EXACTLY:

{{
  "match_percentage": 0,
  "matching_skills": [],
  "missing_skills": [],
  "missing_keywords": [],
  "improvement_suggestions": []
}}
"""


# ============================================================================
# RAG ANSWER
# ============================================================================

RAG_ANSWER_PROMPT = """
Use the retrieved knowledge-base context to answer the candidate's request.

Retrieved context:
---
{context}
---

Candidate context:
---
{candidate_summary}
---

Request:
{query}

IMPORTANT:
- Ground the answer in the retrieved context where relevant.
- Do not pretend that information is in the knowledge base if it is not.
- If the context is insufficient, say that clearly.
- General career knowledge may be used when necessary, but distinguish it
  from knowledge retrieved from the knowledge base.
- Give practical, specific, actionable advice.
- Avoid generic motivational statements.
"""


# ============================================================================
# CAREER ROADMAP
# ============================================================================

CAREER_ROADMAP_PROMPT = """
Build a practical career roadmap using the retrieved context and candidate profile.

Retrieved context:
---
{context}
---

Candidate type:
{candidate_type}

Target role:
{target_role}

Missing skills:
{missing_skills}

IMPORTANT:
- Prioritize skills based on the target role.
- Do not recommend learning everything.
- Focus on the highest-impact skills first.
- Include practical projects, interview preparation, and job-readiness actions
  where appropriate.
- Keep the roadmap realistic.
- Do not promise employment.
- Use the retrieved context where relevant.
- Do not invent resources or claim that a specific course exists unless it is
  supported by the retrieved context.
- Return ONLY valid JSON.

Return EXACTLY:

{{
  "short_term_0_3_months": [],
  "mid_term_3_9_months": [],
  "long_term_9_18_months": [],
  "recommended_resources": []
}}
"""