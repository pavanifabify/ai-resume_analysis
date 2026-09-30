# AI Resume Analyzer using LLM

A production-ready, LLM-powered resume analysis system that behaves like an
experienced technical recruiter. It does **not** treat every candidate the same way —
it first classifies the candidate's career stage (Student, Fresher, Internship
Candidate, Entry Level Professional, Experienced Professional, Career Switcher) and
then calibrates every downstream evaluation to that profile.

## What it does

- Extracts text from PDF and DOCX resumes
- Parses resumes into structured data (contact info, education, skills, experience,
  projects, certifications, achievements)
- Classifies the candidate profile type with a confidence score and reasoning
- Scores the resume (overall + 8 category scores) — calibrated to career stage
- Analyzes skills: present, missing, strength, and recommended skills to learn
- Evaluates each project individually and rewrites weak descriptions
- Evaluates work/internship experience for date consistency and measurable impact
- Matches the resume against a pasted job description (match %, missing skills/keywords)
- Uses a RAG pipeline (FAISS + Sentence-Transformers) over a curated career-guidance
  knowledge base to answer follow-up questions and generate a personalized roadmap
- Persists every analysis to a local SQLite database
- Lets you download the full report as JSON

## Honest limitation

This system uses an LLM for judgment calls (scoring, project evaluation, JD matching).
LLM outputs are not deterministic and can occasionally be wrong or inconsistent between
runs — no resume-analysis tool, including this one, can honestly guarantee 100%
accuracy. To maximize reliability:
- Contact info (email/phone/LinkedIn/GitHub) is extracted with **regex**, not the LLM,
  so those fields are effectively deterministic.
- All LLM outputs are requested as strict JSON and validated/clamped before display.
- Classification uses rule-based signals (graduation year, experience count) as strong
  grounding hints for the LLM, plus a pure rule-based fallback if the LLM call fails.
- Every score is clamped into valid ranges; malformed JSON responses are retried.

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit |
| Backend | Python |
| AI | OpenAI API or Gemini API (configurable) |
| Document Processing | PyMuPDF (PDF), python-docx (DOCX) |
| Database | SQLite |
| Embeddings | Sentence-Transformers (`all-MiniLM-L6-v2`) |
| Vector DB | FAISS |

## Folder Structure

```
resume_analyzer/
├── app.py                      # Streamlit application (entry point)
├── requirements.txt
├── .env.example                 # Copy to .env and fill in your API key
├── README.md
│
├── src/
│   ├── config.py                # Env-based configuration
│   ├── llm_client.py             # Unified OpenAI/Gemini wrapper
│   ├── resume_parser.py          # PDF/DOCX extraction + structured parsing
│   ├── candidate_classifier.py   # Candidate profile classification
│   ├── analyzer.py               # Scoring, skills, projects, experience, JD match
│   ├── embeddings.py             # Sentence-Transformers wrapper
│   ├── rag_pipeline.py           # Chunking, FAISS index, retrieval, RAG generation
│   ├── database.py               # SQLite schema + CRUD
│   └── prompts.py                # All LLM prompt templates
│
├── knowledge_base/               # RAG source documents (.txt / .md / .pdf supported)
│   ├── ats_guidelines.txt
│   ├── resume_tips.txt
│   ├── interview_questions.txt
│   └── skill_roadmaps.txt
│
├── sample_data/
│   ├── sample_resume.txt
│   └── sample_resume.docx        # Ready to upload for a test run
│
├── vector_store/                 # FAISS index cache (auto-created)
└── database/                     # SQLite file (auto-created)
```

## Setup

### 1. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure your API key
```bash
cp .env.example .env
```
Edit `.env`:
- Set `LLM_PROVIDER=openai` (or `gemini`)
- Set `OPENAI_API_KEY=sk-...` (get one at https://platform.openai.com/api-keys)
  — or `GEMINI_API_KEY=...` (get one at https://aistudio.google.com/apikey)

### 4. Run the app
```bash
streamlit run app.py
```
The app opens at `http://localhost:8501`. On first run it will:
- Create the SQLite database at `database/resume_analyzer.db`
- Build the FAISS index from `knowledge_base/` and cache it in `vector_store/`
  (this downloads the embedding model once, ~90MB — subsequent runs are instant)

### 5. Test it
Upload `sample_data/sample_resume.docx` (or `.txt` converted to PDF/DOCX yourself)
and click "Analyze Resume".

## Rebuilding the knowledge base index

If you edit or add files in `knowledge_base/`, delete the cached index so it rebuilds:
```bash
rm -rf vector_store/*
```
The app will automatically rebuild it on next launch.

## Extending

- **Add PDF knowledge base docs**: drop `.pdf` files into `knowledge_base/` — the RAG
  pipeline reads `.txt`, `.md`, and `.pdf` automatically.
- **Switch LLM provider**: change `LLM_PROVIDER` in `.env` — no code changes needed,
  `src/llm_client.py` handles both providers behind the same interface.
- **Add LangChain**: `src/rag_pipeline.py` currently uses FAISS + Sentence-Transformers
  directly for transparency and fewer moving parts; it's straightforward to swap the
  retrieval/generation calls for LangChain's `RetrievalQA` chain if you prefer, since
  `langchain` is already listed in `requirements.txt`.

## Database Schema

- **users** — `id, name, email, created_at`
- **resume_history** — `id, user_id, resume_filename, upload_date, raw_text`
- **analysis_reports** — `id, resume_history_id, candidate_type, overall_score, report_json, created_at`
- **job_descriptions** — `id, resume_history_id, jd_text, match_result_json, created_at`

## Troubleshooting

| Problem | Fix |
|---|---|
| `OPENAI_API_KEY is missing` | Add it to `.env`, not just `.env.example` |
| "Could not extract meaningful text" | Your PDF is likely scanned/image-based — export a text-based PDF instead |
| Slow first run | The embedding model downloads once (~90MB); subsequent runs use the cache |
| `ImportError: openai` / `google.generativeai` | Run `pip install -r requirements.txt` again inside your active venv |
