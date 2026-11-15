# Student Next-Course Recommendation System

An internal academic counselling and programme progression web application built with **Python**, **Streamlit**, **Pandas**, and **Groq** (`openai/gpt-oss-120b`). It enables the counselling and sales team to identify, evaluate, and pitch the top 3 suitable next executive education programmes for students who have completed or are nearing completion of their current programme.

---

## Key Features

- **Automated Data Cleaning & Validation (`src/data_loader.py`, `src/data_validator.py`)**
  - Loads and standardizes `programs.csv` and `students.csv` (with automatic fallback to `students_data.csv`).
  - Trims whitespace, normalizes `enrollment_status` (`completed`, `in_progress`, `withdrawn`), parses multi-value fields (`|` and `;`), and validates duplicate IDs and cross-references.
  - Emits clear warnings if a student's `current_program_code` is missing from `programs.csv` without assuming curriculum details.
- **Eligibility Queue Filtering**
  - Automatically filters the recommendation queue to include only students who have **completed** their programme (`100%`) or are **in progress with $\ge$ 80% completion** (configurable via `MIN_COMPLETION_PERCENTAGE`).
  - Excludes **withdrawn** students and students below 80% completion.
- **Explainable Multi-Factor Recommendation Engine (`src/recommender.py`)**
  - Scores and ranks candidate programmes out of **100** across four configurable weighted dimensions:
    1. **Current Programme Relevance (40%)** — Domain continuity, complementary cross-domain progression, description/audience alignment, and seniority tier progression.
    2. **Professional Relevance (25%)** — Alignment between student designation/industry and programme `target_roles` / `target_industries`.
    3. **Experience & Qualification Fit (20%)** — Evaluates `experience_years` against `minimum_experience_years` as a suitability factor (never hard-rejects) and flags unverifiable free-text prerequisites (e.g., $>50\%$ marks, SOP, MAST/interview) for counsellor verification.
    4. **Skill Progression (15%)** — Evaluates how candidate `skills_covered` reinforce existing foundations while introducing new executive capabilities.
  - Strictly excludes the student's current programme from recommendations.
  - Supports optional **Domain** filtering (`All Domains` or any specific domain from `programs.csv`).
- **Privacy-Safe GenAI Explanations & Pitching Layer (`src/ai_explainer.py`)**
  - Uses `GROQ_API_KEY` and `GROQ_MODEL` (default `openai/gpt-oss-120b`, via `python-dotenv` and Groq's OpenAI-compatible API) to explain the already-ranked top 3 programmes without inventing programmes or altering rankings.
  - **Privacy Guardrails**: Strips all personal identifiers (`student_id`, names, emails, mobile numbers, addresses, demographics) before calling the API.
  - **Graceful Fallback**: Works seamlessly even if `GROQ_API_KEY` is missing or if the Groq API returns an error, providing full catalogue-grounded explanations, **Copy Pitch** buttons, and **Generate Again** / **Retry AI** controls.

---

## Project Structure

```text
course-recommendation-app/
│
├── app.py                        # Main Streamlit application entrypoint (Dashboard & Student Recommendation views)
├── config.py                     # Environment variables, file paths, and scoring weights configuration
├── requirements.txt              # Python package dependencies
├── .env.example                  # Sample environment variables template
├── .gitignore                    # Git ignore rules
├── README.md                     # Documentation and launch guide
│
├── data/
│   ├── programs.csv              # Programme master catalogue
│   └── students_data.csv         # Student academic and professional records (also supports students.csv)
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py            # CSV ingestion, cleaning, and eligibility queue filtering
│   ├── data_validator.py         # Schema validation, duplicate detection, and cross-reference checks
│   ├── recommender.py            # Deterministic 4-factor scoring and ranking engine
│   ├── ai_explainer.py           # Privacy-safe Groq explanation & pitch service with deterministic fallback
│   └── ui_components.py          # Reusable SaaS dashboard cards, filters, tables, and recommendation cards
│
├── assets/
│   └── styles.css                # Custom modern SaaS styling (white cards, slate text, red primary accents)
│
└── tests/
    ├── test_recommender.py       # Unit tests for scoring and ranking logic
    ├── test_ai_explainer.py      # Unit tests for privacy sanitization, Groq handling, and fallbacks
    └── test_scenarios_e2e.py     # End-to-end validation of all 12 operational scenarios
```

---

## Quick Start

### 1. Create and Activate Virtual Environment (First-Time Setup)

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure Environment Variables (Optional for GenAI Layer)

Copy `.env.example` to `.env` and add your Groq API key:

```bash
cp .env.example .env
```

```dotenv
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=openai/gpt-oss-120b
APP_ENV=development
DATA_DIR=data
```

*(Note: The application and recommendation engine work out-of-the-box even if `GROQ_API_KEY` is left empty or unconfigured.)*

### 3. Launch the Application with One Command

```bash
./venv/bin/streamlit run app.py
```

*(Or, if your virtual environment is already activated: `streamlit run app.py`)*

---

## Running the Test Suite

Run all unit and end-to-end scenario tests with:

```bash
./venv/bin/python -m unittest discover -s tests -v
```
