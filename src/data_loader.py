"""Data loading and cleaning module for student and program datasets."""
import re
from pathlib import Path
from typing import List, Optional
import pandas as pd

from config import (
    PROGRAMS_FILE,
    STUDENTS_FILE,
    STUDENTS_DATA_FALLBACK_FILE,
    MIN_COMPLETION_PERCENTAGE,
    RECOMMENDATION_CONFIG,
)


ENROLLMENT_STATUS_MAP = {
    "completed": "completed",
    "complete": "completed",
    "done": "completed",
    "graduated": "completed",
    "in_progress": "in_progress",
    "in progress": "in_progress",
    "in-progress": "in_progress",
    "ongoing": "in_progress",
    "active": "in_progress",
    "enrolled": "in_progress",
    "withdrawn": "withdrawn",
    "withdraw": "withdrawn",
    "dropped": "withdrawn",
    "cancelled": "withdrawn",
    "canceled": "withdrawn",
}


def get_students_csv_path() -> Path:
    """Resolve the student CSV path, supporting both students.csv and students_data.csv."""
    if STUDENTS_FILE.exists():
        return STUDENTS_FILE
    if STUDENTS_DATA_FALLBACK_FILE.exists():
        return STUDENTS_DATA_FALLBACK_FILE
    return STUDENTS_FILE


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Remove leading and trailing spaces from DataFrame column names."""
    df = df.copy()
    df.columns = [str(col).strip() for col in df.columns]
    return df


def clean_text_values(df: pd.DataFrame) -> pd.DataFrame:
    """Strip leading and trailing whitespace from all string/object columns and normalize empty strings to NA."""
    df = df.copy()
    for col in df.columns:
        if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]):
            df[col] = df[col].apply(
                lambda x: x.strip() if isinstance(x, str) else x
            )
            df[col] = df[col].replace({"": pd.NA, "nan": pd.NA, "None": pd.NA, "null": pd.NA})
    return df


def standardize_program_code(code: object) -> Optional[str]:
    """Standardize a programme code by trimming spaces and normalizing internal whitespace."""
    if pd.isna(code):
        return None
    cleaned = re.sub(r"\s+", " ", str(code).strip())
    return cleaned if cleaned else None


def standardize_program_codes_series(series: pd.Series) -> pd.Series:
    """Standardize a pandas Series of programme codes."""
    return series.apply(standardize_program_code)


def standardize_enrollment_status(status: object) -> Optional[str]:
    """Standardize enrollment_status values such as 'completed', 'in_progress', and 'withdrawn'."""
    if pd.isna(status):
        return None
    raw = re.sub(r"\s+", " ", str(status).strip().lower())
    if not raw:
        return None
    if raw in ENROLLMENT_STATUS_MAP:
        return ENROLLMENT_STATUS_MAP[raw]
    normalized_underscores = raw.replace(" ", "_").replace("-", "_")
    return ENROLLMENT_STATUS_MAP.get(normalized_underscores, normalized_underscores)


def parse_multi_value_field(value: object) -> List[str]:
    """Parse fields containing multiple values separated by '|' (or ';') into a clean list of strings.

    Supports pipe '|' primarily as required, and also falls back to ';' when '|' is not present
    (or splits on both '|' and ';') to cleanly handle existing catalog formatting without breaking
    parenthetical expressions.
    """
    if pd.isna(value) or value is None:
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    text = str(value).strip()
    if not text:
        return []

    # Split by pipe '|' or semicolon ';'
    if "|" in text or ";" in text:
        parts = re.split(r"[|;]+", text)
    else:
        # If neither '|' nor ';' is present, check if comma-separated outside parentheses
        parts = re.split(r",(?![^()]*\))", text)

    cleaned_parts = []
    for part in parts:
        item = part.strip()
        if item and item not in cleaned_parts:
            cleaned_parts.append(item)
    return cleaned_parts


def normalize_pipe_separated_string(value: object) -> Optional[str]:
    """Normalize a multi-value string into a canonical '|'-separated string."""
    items = parse_multi_value_field(value)
    return " | ".join(items) if items else None


def convert_numeric_columns(df: pd.DataFrame, numeric_cols: List[str]) -> pd.DataFrame:
    """Convert specified columns to numeric types gracefully without crashing on invalid or missing values."""
    df = df.copy()
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def clean_programs_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and standardize the programs DataFrame."""
    df = clean_column_names(df)
    df = clean_text_values(df)

    # Drop completely empty rows (where all columns are NA)
    df = df.dropna(how="all").reset_index(drop=True)

    if "program_code" in df.columns:
        df["program_code"] = standardize_program_codes_series(df["program_code"])
        # Drop rows where program_code as well as program_name are both missing
        if "program_name" in df.columns:
            df = df.dropna(subset=["program_code", "program_name"], how="all").reset_index(drop=True)

    # Convert numerical fields
    numeric_cols = ["minimum_experience_years", "maximum_experience_years"]
    df = convert_numeric_columns(df, numeric_cols)

    # Support multiple values separated by "|" in roles, industries and skills fields (plus domain)
    multi_value_cols = ["target_roles", "target_industries", "skills_covered", "domain"]
    for col in multi_value_cols:
        if col in df.columns:
            df[f"{col}_list"] = df[col].apply(parse_multi_value_field)

    return df


def clean_students_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and standardize the students DataFrame without discarding incomplete optional records."""
    df = clean_column_names(df)
    df = clean_text_values(df)

    # Drop only completely blank rows (all columns NA)
    df = df.dropna(how="all").reset_index(drop=True)

    if "student_id" in df.columns:
        df["student_id"] = df["student_id"].apply(
            lambda x: str(x).strip() if pd.notna(x) and str(x).strip() else None
        )

    if "current_program_code" in df.columns:
        df["current_program_code"] = standardize_program_codes_series(df["current_program_code"])

    if "enrollment_status" in df.columns:
        df["enrollment_status"] = df["enrollment_status"].apply(standardize_enrollment_status)

    # Convert numerical fields
    numeric_cols = ["completion_percentage", "completion_year", "experience_years"]
    df = convert_numeric_columns(df, numeric_cols)

    # Support pipe-separated multi-values in student industry/designation fields if present
    student_multi_cols = ["current_industry", "previous_industry", "designation"]
    for col in student_multi_cols:
        if col in df.columns:
            df[f"{col}_list"] = df[col].apply(parse_multi_value_field)

    return df


def is_student_eligible_for_recommendation(
    student_row: pd.Series,
    min_completion_percentage: Optional[float] = None,
) -> bool:
    """Determine if a single student record is eligible for the recommendation queue.

    Rules:
    - Do not recommend courses to withdrawn students.
    - Eligible when enrollment_status is 'completed', OR
    - enrollment_status is 'in_progress' AND completion_percentage >= min_completion_percentage (default 80%).
    """
    if min_completion_percentage is None:
        min_completion_percentage = float(
            RECOMMENDATION_CONFIG.get("min_completion_threshold", MIN_COMPLETION_PERCENTAGE)
        )

    status = standardize_enrollment_status(student_row.get("enrollment_status"))
    if not status or status == "withdrawn":
        return False

    if status == "completed":
        return True

    if status == "in_progress":
        comp_pct = student_row.get("completion_percentage")
        if pd.isna(comp_pct):
            return False
        try:
            return float(comp_pct) >= min_completion_percentage
        except (ValueError, TypeError):
            return False

    return False


def get_recommendation_queue(
    students_df: pd.DataFrame,
    min_completion_percentage: Optional[float] = None,
) -> pd.DataFrame:
    """Filter and return the DataFrame of students eligible for the recommendation queue.

    Also adds an `is_eligible` boolean column and `eligibility_reason` explanation column to a copy.
    """
    if min_completion_percentage is None:
        min_completion_percentage = float(
            RECOMMENDATION_CONFIG.get("min_completion_threshold", MIN_COMPLETION_PERCENTAGE)
        )

    if students_df.empty:
        return students_df.copy()

    mask = students_df.apply(
        lambda row: is_student_eligible_for_recommendation(
            row, min_completion_percentage=min_completion_percentage
        ),
        axis=1,
    )
    return students_df[mask].copy().reset_index(drop=True)


def annotate_student_eligibility(
    students_df: pd.DataFrame,
    min_completion_percentage: Optional[float] = None,
) -> pd.DataFrame:
    """Return a copy of students_df with `is_eligible` and `ineligibility_reason` columns."""
    if min_completion_percentage is None:
        min_completion_percentage = float(
            RECOMMENDATION_CONFIG.get("min_completion_threshold", MIN_COMPLETION_PERCENTAGE)
        )

    df = students_df.copy()
    eligible_flags = []
    reasons = []

    for _, row in df.iterrows():
        status = standardize_enrollment_status(row.get("enrollment_status"))
        comp_pct = row.get("completion_percentage")

        if status == "withdrawn":
            eligible_flags.append(False)
            reasons.append("Withdrawn students are excluded from recommendations.")
        elif status == "completed":
            eligible_flags.append(True)
            reasons.append("Eligible (Completed current programme).")
        elif status == "in_progress":
            if pd.notna(comp_pct) and float(comp_pct) >= min_completion_percentage:
                eligible_flags.append(True)
                reasons.append(
                    f"Eligible (In progress with {float(comp_pct):.0f}% >= {min_completion_percentage:.0f}% threshold)."
                )
            else:
                pct_str = f"{float(comp_pct):.0f}%" if pd.notna(comp_pct) else "missing"
                eligible_flags.append(False)
                reasons.append(
                    f"In progress completion ({pct_str}) is below {min_completion_percentage:.0f}% threshold."
                )
        else:
            eligible_flags.append(False)
            reasons.append(f"Unrecognized or missing enrollment status: {status!r}.")

    df["is_eligible"] = eligible_flags
    df["eligibility_reason"] = reasons
    return df


def load_programs_data(file_path: Optional[Path] = None, clean: bool = True) -> pd.DataFrame:
    """Load programs CSV into a pandas DataFrame and optionally clean/standardize it."""
    path = Path(file_path) if file_path else PROGRAMS_FILE
    if not path.exists():
        raise FileNotFoundError(f"Programs data file not found at: {path}")
    df = pd.read_csv(path)
    if clean:
        df = clean_programs_dataframe(df)
    return df


def load_students_data(file_path: Optional[Path] = None, clean: bool = True) -> pd.DataFrame:
    """Load students CSV into a pandas DataFrame and optionally clean/standardize it."""
    path = Path(file_path) if file_path else get_students_csv_path()
    if not path.exists():
        raise FileNotFoundError(f"Students data file not found at: {path}")
    df = pd.read_csv(path)
    if clean:
        df = clean_students_dataframe(df)
    return df
