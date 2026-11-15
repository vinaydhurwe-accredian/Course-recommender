import os
from pathlib import Path
from dotenv import load_dotenv

# Base Directory
BASE_DIR = Path(__file__).resolve().parent

# Load environment variables from .env if present
load_dotenv(dotenv_path=BASE_DIR / ".env")

# Environment & API Settings
APP_ENV = os.getenv("APP_ENV", "development")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
GROQ_BASE_URL = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")

# Data File Paths
DATA_DIR = BASE_DIR / os.getenv("DATA_DIR", "data")
PROGRAMS_FILE = DATA_DIR / "programs.csv"
STUDENTS_FILE = DATA_DIR / "students.csv"
# Fallback support if students file is named students_data.csv
STUDENTS_DATA_FALLBACK_FILE = DATA_DIR / "students_data.csv"

# Recommendation System Default Configurations
MIN_COMPLETION_PERCENTAGE = float(os.getenv("MIN_COMPLETION_PERCENTAGE", "80.0"))

RECOMMENDATION_CONFIG = {
    # Default number of program recommendations to return
    "top_k": 3,
    # Minimum completion percentage required for in_progress students to be eligible for next recommendations
    "min_completion_threshold": MIN_COMPLETION_PERCENTAGE,
    # Eligible student enrollment statuses
    "eligible_statuses": ["completed", "in_progress"],
    # Disqualified student enrollment statuses
    "excluded_statuses": ["withdrawn"],
    # Scoring weights for ranking candidates (Total = 1.00 / 100%)
    "weights": {
        "current_program_relevance": 0.40,
        "professional_relevance": 0.25,
        "experience_qualification_fit": 0.20,
        "skill_progression": 0.15,
    },
    # AI Explanation settings
    "ai_explanation": {
        "enabled": True,
        "temperature": 0.3,
        "max_tokens": 500,
    },
}

