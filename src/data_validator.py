"""Data validation module to verify integrity, schemas, duplicates, and cross-references of datasets."""
import warnings
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd

from config import MIN_COMPLETION_PERCENTAGE, RECOMMENDATION_CONFIG
from src.data_loader import (
    annotate_student_eligibility,
    get_recommendation_queue,
    standardize_program_code,
)

REQUIRED_PROGRAM_COLUMNS = [
    "program_code",
    "program_name",
    "domain",
    "program_description",
    "target_audience",
    "target_roles",
    "target_industries",
    "skills_covered",
    "prerequisites",
    "program_level",
    "minimum_experience_years",
    "maximum_experience_years",
    "qualification_requirements",
    "duration",
]

REQUIRED_STUDENT_COLUMNS = [
    "student_id",
    "current_program_code",
    "enrollment_status",
    "completion_percentage",
    "qualification",
    "degree_name",
    "completion_year",
    "experience_years",
    "designation",
    "working_status",
    "current_industry",
    "previous_industry",
]


def validate_required_columns(
    df: pd.DataFrame, required_columns: List[str], dataset_name: str = "dataset"
) -> Tuple[bool, List[str]]:
    """Validate that all required columns are present in the DataFrame."""
    actual_cols = [str(col).strip() for col in df.columns]
    missing_cols = [col for col in required_columns if col not in actual_cols]
    errors = []
    if missing_cols:
        errors.append(f"Missing required columns in {dataset_name}: {missing_cols}")
    return len(errors) == 0, errors


def find_duplicate_program_codes(programs_df: pd.DataFrame) -> List[str]:
    """Identify duplicate programme codes in programs_df."""
    if "program_code" not in programs_df.columns or programs_df.empty:
        return []
    codes = programs_df["program_code"].dropna().apply(standardize_program_code).dropna()
    duplicates = codes[codes.duplicated(keep=False)].unique().tolist()
    return duplicates


def find_duplicate_student_ids(students_df: pd.DataFrame) -> List[str]:
    """Identify duplicate student IDs in students_df."""
    if "student_id" not in students_df.columns or students_df.empty:
        return []
    ids = students_df["student_id"].dropna().astype(str).str.strip()
    ids = ids[ids != ""]
    duplicates = ids[ids.duplicated(keep=False)].unique().tolist()
    return duplicates


def find_students_with_missing_program_in_catalog(
    students_df: pd.DataFrame,
    programs_df: pd.DataFrame,
    emit_warning: bool = True,
) -> List[Dict[str, Any]]:
    """Identify students whose current_program_code is not present in programs.csv.

    If a student's current programme is missing from the programme catalogue, displays a warning
    and does not silently assume what that programme contains.
    """
    if (
        students_df.empty
        or "current_program_code" not in students_df.columns
        or "program_code" not in programs_df.columns
    ):
        return []

    valid_program_codes = set(
        programs_df["program_code"]
        .dropna()
        .apply(standardize_program_code)
        .dropna()
        .unique()
    )

    unmatched_records: List[Dict[str, Any]] = []
    for idx, row in students_df.iterrows():
        student_id = row.get("student_id", f"Row_{idx}")
        raw_code = row.get("current_program_code")
        clean_code = standardize_program_code(raw_code)

        if not clean_code or clean_code not in valid_program_codes:
            warning_msg = (
                f"WARNING: Student '{student_id}' has current_program_code "
                f"'{clean_code or 'MISSING'}' which is not present in the programme catalogue (programs.csv). "
                "Programme details will not be assumed."
            )
            unmatched_records.append(
                {
                    "student_id": student_id,
                    "current_program_code": clean_code,
                    "warning": warning_msg,
                }
            )
            if emit_warning:
                warnings.warn(warning_msg, UserWarning, stacklevel=2)

    return unmatched_records


def find_incomplete_optional_student_records(students_df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Identify student records that have missing optional fields without discarding them."""
    optional_cols = [
        "qualification",
        "degree_name",
        "completion_year",
        "experience_years",
        "designation",
        "working_status",
        "current_industry",
        "previous_industry",
    ]
    present_optional_cols = [c for c in optional_cols if c in students_df.columns]
    incomplete_records = []

    for idx, row in students_df.iterrows():
        missing_fields = [col for col in present_optional_cols if pd.isna(row.get(col))]
        if missing_fields:
            incomplete_records.append(
                {
                    "student_id": row.get("student_id", f"Row_{idx}"),
                    "missing_fields": missing_fields,
                }
            )

    return incomplete_records


def validate_programs_dataframe(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """Validate program DataFrame structure, required columns, and duplicate program codes."""
    errors: List[str] = []
    cols_ok, col_errors = validate_required_columns(df, REQUIRED_PROGRAM_COLUMNS, "programs data")
    errors.extend(col_errors)

    if df.empty:
        errors.append("Programs dataset is empty.")
        return False, errors

    if cols_ok:
        dup_codes = find_duplicate_program_codes(df)
        if dup_codes:
            errors.append(f"Duplicate programme codes found in programs data: {dup_codes}")

        missing_code_count = int(df["program_code"].isna().sum())
        if missing_code_count > 0:
            errors.append(f"Found {missing_code_count} programme row(s) with missing program_code.")

    return len(errors) == 0, errors


def validate_students_dataframe(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """Validate student DataFrame structure, required columns, and duplicate student IDs."""
    errors: List[str] = []
    cols_ok, col_errors = validate_required_columns(df, REQUIRED_STUDENT_COLUMNS, "students data")
    errors.extend(col_errors)

    if df.empty:
        errors.append("Students dataset is empty.")
        return False, errors

    if cols_ok:
        dup_ids = find_duplicate_student_ids(df)
        if dup_ids:
            errors.append(f"Duplicate student IDs found in students data: {dup_ids}")

        missing_id_count = int(df["student_id"].isna().sum())
        if missing_id_count > 0:
            errors.append(f"Found {missing_id_count} student row(s) with missing student_id.")

    return len(errors) == 0, errors


def validate_all_datasets(
    students_df: pd.DataFrame,
    programs_df: pd.DataFrame,
    min_completion_percentage: Optional[float] = None,
    emit_warning: bool = False,
) -> Dict[str, Any]:
    """Perform comprehensive validation across both student and program datasets.

    Returns a structured report containing:
    - Schema validation results
    - Duplicate checks for student_id and program_code
    - Cross-reference check for student current_program_code in programs.csv
    - Incomplete optional fields report (retained without discarding)
    - Recommendation queue eligibility summary
    """
    if min_completion_percentage is None:
        min_completion_percentage = float(
            RECOMMENDATION_CONFIG.get("min_completion_threshold", MIN_COMPLETION_PERCENTAGE)
        )

    programs_valid, program_errors = validate_programs_dataframe(programs_df)
    students_valid, student_errors = validate_students_dataframe(students_df)

    duplicate_program_codes = find_duplicate_program_codes(programs_df)
    duplicate_student_ids = find_duplicate_student_ids(students_df)

    unmatched_programs = find_students_with_missing_program_in_catalog(
        students_df, programs_df, emit_warning=emit_warning
    )
    incomplete_student_records = find_incomplete_optional_student_records(students_df)

    annotated_students = annotate_student_eligibility(
        students_df, min_completion_percentage=min_completion_percentage
    )
    eligible_queue = get_recommendation_queue(
        students_df, min_completion_percentage=min_completion_percentage
    )

    warnings_list: List[str] = [item["warning"] for item in unmatched_programs]
    if incomplete_student_records:
        for rec in incomplete_student_records:
            warnings_list.append(
                f"INFO/WARNING: Student '{rec['student_id']}' has incomplete optional fields "
                f"{rec['missing_fields']} (record retained)."
            )

    status_counts = (
        students_df["enrollment_status"].value_counts(dropna=False).to_dict()
        if "enrollment_status" in students_df.columns
        else {}
    )

    return {
        "is_valid": programs_valid and students_valid,
        "programs_valid": programs_valid,
        "students_valid": students_valid,
        "program_errors": program_errors,
        "student_errors": student_errors,
        "total_programs": len(programs_df),
        "total_students": len(students_df),
        "duplicate_program_codes": duplicate_program_codes,
        "duplicate_student_ids": duplicate_student_ids,
        "unmatched_student_programs": unmatched_programs,
        "incomplete_optional_students": incomplete_student_records,
        "enrollment_status_counts": status_counts,
        "min_completion_threshold": min_completion_percentage,
        "eligible_students_count": len(eligible_queue),
        "ineligible_students_count": len(students_df) - len(eligible_queue),
        "eligible_queue_df": eligible_queue,
        "annotated_students_df": annotated_students,
        "warnings": warnings_list,
    }
