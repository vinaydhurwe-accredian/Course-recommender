"""GenAI Explanation and Pitching Service using Groq API (OpenAI-compatible endpoint).

Strictly acts as an explanation and communication layer on top of the Python
recommendation engine. Never computes scores, never changes rankings, and never
invents programmes or attributes outside the supplied profile and catalogue data.
"""
import json
import os
import re
from typing import Any, Dict, List, Optional
import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from config import BASE_DIR, RECOMMENDATION_CONFIG
from src.data_loader import standardize_program_code


# Privacy: Explicit blocklist of PII, personal identifiers, and demographic fields
FORBIDDEN_PII_SUBSTRINGS = (
    "student_id",
    "name",
    "first_name",
    "last_name",
    "full_name",
    "email",
    "mail",
    "phone",
    "mobile",
    "contact",
    "address",
    "dob",
    "date_of_birth",
    "birth",
    "age",
    "gender",
    "sex",
    "caste",
    "religion",
    "race",
    "ethnicity",
    "nationality",
    "aadhaar",
    "pan",
    "passport",
    "ssn",
    "id_number",
    "roll_no",
    "registration_no",
)

# Minimum necessary academic and professional fields permitted from a student record
ALLOWED_STUDENT_FIELDS = (
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
)

# Permitted programme catalogue and recommendation fields
ALLOWED_PROGRAM_CATALOG_FIELDS = (
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
)

PLACEHOLDER_API_KEYS = {
    "",
    "your_groq_api_key_here",
    "your_openai_api_key_here",
    "your-api-key-here",
    "gsk_your-api-key-here",
    "sk-your-api-key-here",
    "none",
    "null",
    "changeme",
    "replace_me",
}


class ProgramAIExplanationSchema(BaseModel):
    """Structured schema for a single recommended programme's AI explanation."""

    program_code: str = Field(
        description="Exact program_code of the recommended programme being explained."
    )
    why_this_programme: str = Field(
        description=(
            "1. Why this programme? Explain why the recommended programme is relevant "
            "to the student's existing academic and professional background."
        )
    )
    learning_progression: str = Field(
        description=(
            "2. Learning progression: Explain how the recommended programme builds on "
            "or expands the student's current learning."
        )
    )
    professional_relevance: str = Field(
        description=(
            "3. Professional relevance: Explain how the programme relates to their "
            "current designation and industry."
        )
    )
    suggested_pitch: str = Field(
        description=(
            "4. Suggested pitch: A short, natural pitching angle that the counselling "
            "team can use while contacting the student (never use personal names)."
        )
    )
    points_to_verify: List[str] = Field(
        description=(
            "5. Points to verify: List of prerequisites, eligibility conditions, or "
            "missing student information that a counsellor should verify before pitching."
        )
    )


class BatchProgramAIExplanationsSchema(BaseModel):
    """Top-level structured schema for the top recommended programmes."""

    explanations: List[ProgramAIExplanationSchema]


SYSTEM_INSTRUCTIONS = """You are an internal academic counselling assistant for Accredian's executive education team.
Your ONLY role is to explain and communicate existing programme recommendations that have already been calculated and ranked by the deterministic Python recommendation engine.

STRICT RULES YOU MUST FOLLOW:
1. DO NOT change the ranking, alter any scores, or invent new programmes.
2. USE ONLY the factual information supplied in the student academic/professional profile and the shortlisted programme catalogue entries.
3. NEVER invent programme features, durations, eligibility criteria, fees, discounts, scholarships, or placement outcomes. If fees or placement details are not in the data, do not mention or fabricate them.
4. NEVER make guaranteed career transition, promotion, job placement, or salary claims.
5. CLEARLY DISTINGUISH known facts (from the supplied student profile and programme data) from assumptions or inferences (e.g., if a field is "Not Provided" or if the current programme is not found in the catalogue, explicitly state what is known vs. what is assumed or requires verification).
6. USE simple, clear, objective professional language.
7. AVOID generic marketing language, hype, buzzwords, or exaggerated sales superlatives (e.g., avoid "game-changing", "world-class", "skyrocket your career", "unlock limitless potential", "transformative journey").
8. DO NOT invent or use any student name or personal identifier; refer to the learner simply as "the student" or "you" (in the pitch).

For each recommended programme provided in the input, return a JSON object matching this exact structure:
{
  "explanations": [
    {
      "program_code": "<exact program_code>",
      "why_this_programme": "<2-4 sentences explaining why this programme is relevant to the student's existing academic qualification, current programme, and professional background. Clearly note known profile facts vs. any assumptions.>",
      "learning_progression": "<2-4 sentences explaining how the programme's curriculum and skills covered build on or expand the student's current learning in their existing programme and degree.>",
      "professional_relevance": "<2-4 sentences explaining how the programme relates to the student's current designation, years of experience, and current/previous industry against the programme's target roles and target industries.>",
      "suggested_pitch": "<A short, natural, conversational 2-3 sentence pitching angle that an academic counsellor can directly use when speaking with the student, grounded strictly in their current programme/role and the recommended programme's actual curriculum and duration.>",
      "points_to_verify": [
        "<Specific prerequisite, eligibility requirement (e.g., minimum 50% marks, required years of experience, SOP, entrance test/interview), or missing student profile field that the counsellor must verify before pitching>"
      ]
    }
  ]
}"""


def _clean_scalar(val: Any, fallback: str = "Not Provided") -> Any:
    """Convert pandas/numpy scalars or missing values into clean JSON-serializable values."""
    if pd.isna(val) or val is None:
        return fallback
    if isinstance(val, float):
        if val.is_integer():
            return int(val)
        return round(val, 2)
    text = str(val).strip()
    if not text or text.lower() in {"nan", "none", "null", "<na>"}:
        return fallback
    return text


def is_Forbidden_key(key: str) -> bool:
    """Return True if a dictionary key matches any forbidden PII or personal identifier pattern."""
    k_lower = str(key).strip().lower()
    if k_lower in ALLOWED_STUDENT_FIELDS or k_lower in ALLOWED_PROGRAM_CATALOG_FIELDS:
        return False
    return any(token in k_lower for token in FORBIDDEN_PII_SUBSTRINGS)


def sanitize_student_profile_for_ai(
    student_record: Any,
    programs_df: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """Extract ONLY the minimum necessary academic and professional fields for the AI prompt.

    Explicitly excludes student_id, student names, emails, mobile numbers, personal identifiers,
    and sensitive demographic information.
    """
    if isinstance(student_record, pd.Series):
        raw_dict = student_record.to_dict()
    elif isinstance(student_record, dict):
        raw_dict = dict(student_record)
    else:
        raw_dict = {}

    sanitized_profile: Dict[str, Any] = {}
    missing_fields: List[str] = []

    for field in ALLOWED_STUDENT_FIELDS:
        cleaned_val = _clean_scalar(raw_dict.get(field), fallback="Not Provided")
        sanitized_profile[field] = cleaned_val
        if cleaned_val == "Not Provided":
            missing_fields.append(field)

    # Attach factual catalogue details of the student's current programme if present in programs_df
    curr_code = standardize_program_code(raw_dict.get("current_program_code"))
    current_prog_details: Optional[Dict[str, Any]] = None
    if curr_code and programs_df is not None and not programs_df.empty and "program_code" in programs_df.columns:
        matched = programs_df[
            programs_df["program_code"].apply(standardize_program_code) == curr_code
        ]
        if not matched.empty:
            prog_row = matched.iloc[0]
            current_prog_details = {
                "program_code": _clean_scalar(prog_row.get("program_code")),
                "program_name": _clean_scalar(prog_row.get("program_name")),
                "domain": _clean_scalar(prog_row.get("domain")),
                "program_level": _clean_scalar(prog_row.get("program_level")),
                "duration": _clean_scalar(prog_row.get("duration")),
                "skills_covered": _clean_scalar(prog_row.get("skills_covered")),
                "program_description": _clean_scalar(prog_row.get("program_description")),
            }

    if current_prog_details is not None:
        sanitized_profile["current_program_catalog_details"] = current_prog_details
    else:
        sanitized_profile["current_program_catalog_details"] = (
            f"Current programme '{curr_code or 'Not Provided'}' is NOT present in programs.csv. "
            "Do not assume its curriculum or skills."
        )
        missing_fields.append("current_program_in_catalog")

    sanitized_profile["missing_or_unverified_student_fields"] = missing_fields
    return sanitized_profile


def sanitize_shortlisted_programs_for_ai(
    recommendations: List[Dict[str, Any]],
    programs_df: Optional[pd.DataFrame] = None,
) -> List[Dict[str, Any]]:
    """Extract only the shortlisted recommended programmes and their factual catalogue attributes."""
    shortlisted_payload: List[Dict[str, Any]] = []

    for idx, rec in enumerate(recommendations, start=1):
        prog_code = standardize_program_code(rec.get("program_code")) or str(rec.get("program_code", "")).strip()

        catalog_info: Dict[str, Any] = {}
        if prog_code and programs_df is not None and not programs_df.empty and "program_code" in programs_df.columns:
            matched = programs_df[
                programs_df["program_code"].apply(standardize_program_code) == prog_code
            ]
            if not matched.empty:
                row = matched.iloc[0]
                for col in ALLOWED_PROGRAM_CATALOG_FIELDS:
                    catalog_info[col] = _clean_scalar(row.get(col))

        # Fall back to fields already in the recommendation dict if not looked up from DataFrame
        for col in ALLOWED_PROGRAM_CATALOG_FIELDS:
            if col not in catalog_info and col in rec:
                catalog_info[col] = _clean_scalar(rec.get(col))

        shortlisted_payload.append(
            {
                "rank": int(rec.get("rank", idx)),
                "program_code": prog_code,
                "program_name": catalog_info.get("program_name", _clean_scalar(rec.get("program_name"))),
                "domain": catalog_info.get("domain", _clean_scalar(rec.get("domain"))),
                "program_level": catalog_info.get("program_level", _clean_scalar(rec.get("program_level"))),
                "duration": catalog_info.get("duration", _clean_scalar(rec.get("duration"))),
                "program_description": catalog_info.get("program_description", "Not Provided"),
                "target_audience": catalog_info.get("target_audience", "Not Provided"),
                "target_roles": catalog_info.get("target_roles", "Not Provided"),
                "target_industries": catalog_info.get("target_industries", "Not Provided"),
                "skills_covered": catalog_info.get("skills_covered", "Not Provided"),
                "prerequisites": catalog_info.get("prerequisites", _clean_scalar(rec.get("prerequisites"))),
                "minimum_experience_years": catalog_info.get(
                    "minimum_experience_years", _clean_scalar(rec.get("minimum_experience_years"))
                ),
                "maximum_experience_years": catalog_info.get(
                    "maximum_experience_years", _clean_scalar(rec.get("maximum_experience_years"))
                ),
                "qualification_requirements": catalog_info.get(
                    "qualification_requirements", _clean_scalar(rec.get("qualification_requirements"))
                ),
                "engine_scores": {
                    "overall_match_score": rec.get("overall_match_score"),
                    "current_program_relevance_score": rec.get("current_program_relevance_score"),
                    "professional_relevance_score": rec.get("professional_relevance_score"),
                    "experience_qualification_score": rec.get("experience_qualification_score"),
                    "skill_progression_score": rec.get("skill_progression_score"),
                },
                "engine_evidence_reasons": list(rec.get("match_reasons", [])),
                "engine_identified_gaps_and_verifications": list(rec.get("potential_gaps", [])),
            }
        )

    return shortlisted_payload


def build_privacy_safe_api_payload(
    student_record: Any,
    recommendations: List[Dict[str, Any]],
    programs_df: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """Construct the complete privacy-safe JSON payload sent to Groq."""
    return {
        "student_academic_and_professional_profile": sanitize_student_profile_for_ai(
            student_record=student_record,
            programs_df=programs_df,
        ),
        "shortlisted_recommended_programmes": sanitize_shortlisted_programs_for_ai(
            recommendations=recommendations,
            programs_df=programs_df,
        ),
    }


def _redact_student_ids_from_payload(payload: Dict[str, Any], student_id: Optional[str] = None) -> Dict[str, Any]:
    """Ensure any accidental mention of a student_id inside engine warning/gap strings is scrubbed."""
    raw_json = json.dumps(payload, ensure_ascii=False)
    if student_id and str(student_id).strip():
        raw_json = raw_json.replace(str(student_id).strip(), "the student")
    # Also scrub generic STU### patterns
    raw_json = re.sub(r"\bSTU[_A-Za-z0-9]+\b", "the student", raw_json)
    return json.loads(raw_json)


def build_deterministic_explanation_for_program(
    student_record: Any,
    rec: Dict[str, Any],
    programs_df: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """Build a factual, catalogue-grounded 5-part counselling explanation and pitch
    when Groq is unconfigured or unavailable, so every recommendation card always displays
    all required fields without inventing facts.
    """
    profile = sanitize_student_profile_for_ai(student_record, programs_df=programs_df)
    shortlisted = sanitize_shortlisted_programs_for_ai([rec], programs_df=programs_df)
    prog_info = shortlisted[0] if shortlisted else {}

    prog_code = str(rec.get("program_code", "")).strip()
    prog_name = str(rec.get("program_name", prog_code)).strip()
    domain = str(rec.get("domain", "Not Specified")).strip()
    duration = str(rec.get("duration", "Not Specified")).strip()
    level = str(rec.get("program_level", "Not Specified")).strip()
    prereqs = str(rec.get("prerequisites", "No specific prerequisites stated in catalogue.")).strip()

    curr_code = str(profile.get("current_program_code", "Not Provided"))
    qual = str(profile.get("qualification", "Not Provided"))
    degree = str(profile.get("degree_name", "Not Provided"))
    exp_years = profile.get("experience_years", "Not Provided")
    designation = str(profile.get("designation", "Not Provided"))
    curr_ind = str(profile.get("current_industry", "Not Provided"))
    prev_ind = str(profile.get("previous_industry", "Not Provided"))

    curr_prog_cat = profile.get("current_program_catalog_details")
    has_curr_cat = isinstance(curr_prog_cat, dict)
    curr_prog_label = (
        f"{curr_code} ({curr_prog_cat.get('program_name')})"
        if has_curr_cat and curr_prog_cat.get("program_name")
        else curr_code
    )

    reasons: List[str] = [str(r) for r in rec.get("match_reasons", []) if r]
    gaps: List[str] = [str(g) for g in rec.get("potential_gaps", []) if g]

    # 1. Why it matches the student
    known_bg_parts: List[str] = []
    if degree != "Not Provided":
        known_bg_parts.append(f"academic background in {degree}")
    elif qual != "Not Provided":
        known_bg_parts.append(f"{qual} qualification")
    if exp_years != "Not Provided":
        known_bg_parts.append(f"{exp_years} years of professional experience")
    if designation != "Not Provided":
        known_bg_parts.append(f"current role as {designation}")

    bg_summary = ", ".join(known_bg_parts) if known_bg_parts else "available profile signals"
    reason_summary = " ".join(reasons[:2]) if reasons else f"Aligns with {domain} at the {level} tier."
    missing_fields = profile.get("missing_or_unverified_student_fields", [])
    assumption_note = (
        f" Note: {', '.join(missing_fields[:3])} is not provided in the student record and should be confirmed."
        if missing_fields
        else ""
    )
    why_this = (
        f"{prog_name} ({prog_code}) is recommended based on the student's {bg_summary}. "
        f"{reason_summary}{assumption_note}"
    ).strip()

    # 2. How it builds on their existing course (Learning progression)
    cand_skills_raw = str(prog_info.get("skills_covered", "Not Provided"))
    cand_skills_preview = ", ".join(
        [s.strip() for s in re.split(r"[|;]+", cand_skills_raw) if s.strip()][:5]
    )
    if has_curr_cat:
        curr_dom = curr_prog_cat.get("domain", "their current domain")
        learning_prog = (
            f"Having undertaken {curr_prog_label} in {curr_dom}, the student can progress into "
            f"{prog_name} ({domain}, {level}) to expand capabilities in "
            f"{cand_skills_preview or domain}."
        )
    else:
        learning_prog = (
            f"The student's current programme ({curr_code}) is not listed in programs.csv, so no assumptions "
            f"are made about its syllabus. Based on catalogue data, {prog_name} ({level}, {duration}) "
            f"develops structured capabilities in {cand_skills_preview or domain}."
        )

    # 3. Professional relevance
    target_roles_raw = str(prog_info.get("target_roles", "Not Provided"))
    target_roles_preview = ", ".join(
        [r.strip() for r in re.split(r"[|;]+", target_roles_raw) if r.strip()][:4]
    )
    ind_context = (
        f"the {curr_ind} sector"
        if curr_ind != "Not Provided"
        else (f"their previous {prev_ind} industry background" if prev_ind != "Not Provided" else "their industry")
    )
    role_context = f"as a {designation}" if designation != "Not Provided" else "in their current professional capacity"
    prof_rel = (
        f"Relevant to the student's work {role_context} in {ind_context}. "
        f"In the catalogue, {prog_code} specifically targets roles such as {target_roles_preview or 'executive and managerial leaders'}."
    )

    # 4. Suggested counsellor pitch
    pitch_opening = (
        f"Since you have been progressing through {curr_prog_label}"
        if curr_code != "Not Provided"
        else "Based on your academic and professional background"
    )
    pitch_role = (
        f" and working as {designation}"
        if designation != "Not Provided"
        else ""
    )
    suggested_pitch = (
        f"{pitch_opening}{pitch_role}, {prog_name} ({prog_code}) is a natural next step to deepen your expertise in {domain}. "
        f"This {duration} ({level}) programme covers {cand_skills_preview or 'core strategic and leadership modules'} "
        f"that align directly with your career profile."
    )

    # 5. Prerequisites or points to verify
    points_to_verify: List[str] = []
    if prereqs:
        points_to_verify.append(f"Catalogue Prerequisites: {prereqs}")
    for g in gaps:
        if g not in points_to_verify:
            points_to_verify.append(g)
    for mf in missing_fields:
        label_mf = mf.replace("_", " ")
        msg = f"Verify missing student profile field: {label_mf}."
        if msg not in points_to_verify:
            points_to_verify.append(msg)

    return {
        "program_code": prog_code,
        "why_this_programme": why_this,
        "learning_progression": learning_prog,
        "professional_relevance": prof_rel,
        "suggested_pitch": suggested_pitch,
        "points_to_verify": points_to_verify,
    }


class AIExplainer:
    """Generates grounded, privacy-safe AI explanations and counselling pitches via Groq."""

    DEFAULT_GROQ_BASE_URL = "https://api.groq.com/openai/v1"
    DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        client: Optional[Any] = None,
    ):
        # Reload .env dynamically so updates to .env take effect without restarting
        load_dotenv(dotenv_path=BASE_DIR / ".env", override=False)

        raw_key = api_key if api_key is not None else os.getenv("GROQ_API_KEY", "")
        raw_model = model if model is not None else os.getenv("GROQ_MODEL", self.DEFAULT_GROQ_MODEL)
        raw_base_url = base_url if base_url is not None else os.getenv("GROQ_BASE_URL", self.DEFAULT_GROQ_BASE_URL)

        self.api_key = str(raw_key).strip()
        self.model = str(raw_model).strip() or self.DEFAULT_GROQ_MODEL
        self.base_url = str(raw_base_url).strip() or self.DEFAULT_GROQ_BASE_URL
        self._client = client

        ai_cfg = (RECOMMENDATION_CONFIG or {}).get("ai_explanation", {})
        self.temperature = float(ai_cfg.get("temperature", 0.2))
        self.max_tokens = int(ai_cfg.get("max_tokens", 1800))

    def is_configured(self) -> bool:
        """Return True if a non-placeholder Groq API key is configured."""
        if not self.api_key:
            return False
        if self.api_key.lower() in PLACEHOLDER_API_KEYS:
            return False
        return True

    def _get_client(self) -> Any:
        """Lazily initialize and return the Groq OpenAI-compatible client."""
        if self._client is not None:
            return self._client
        from openai import OpenAI

        self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        return self._client

    def generate_explanations_for_recommendations(
        self,
        student_record: Any,
        recommendations: List[Dict[str, Any]],
        programs_df: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """Generate structured AI explanations for the top recommended programmes.

        Never raises an unhandled exception: returns a structured result dictionary with:
        - `success` (bool)
        - `model` (str)
        - `explanations` (Dict[program_code, Dict[str, Any]])
        - `error` (Optional[str])
        - `error_type` (Optional[str]: 'missing_key' | 'api_error' | 'parse_error')
        """
        if not recommendations:
            return {
                "success": False,
                "model": self.model,
                "explanations": {},
                "error": "No recommended programmes available to explain.",
                "error_type": "no_recommendations",
            }

        if not self.is_configured():
            return {
                "success": False,
                "model": self.model,
                "explanations": {},
                "error": (
                    "Groq API key is missing or set to the placeholder value. "
                    "Please set a valid GROQ_API_KEY in your .env file to enable AI explanations. "
                    "The deterministic recommendation engine and scores above remain fully functional."
                ),
                "error_type": "missing_key",
            }

        student_id = None
        if isinstance(student_record, (pd.Series, dict)):
            student_id = student_record.get("student_id")

        raw_payload = build_privacy_safe_api_payload(
            student_record=student_record,
            recommendations=recommendations,
            programs_df=programs_df,
        )
        safe_payload = _redact_student_ids_from_payload(raw_payload, student_id=student_id)

        user_prompt = (
            "Generate the 5-part counselling explanation and pitch for each of the following "
            "shortlisted programme recommendations based strictly on the supplied student profile "
            "and programme catalogue data.\n\n"
            f"INPUT DATA (JSON):\n{json.dumps(safe_payload, indent=2, ensure_ascii=False)}"
        )

        try:
            client = self._get_client()
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_INSTRUCTIONS},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
                temperature=self.temperature,
                max_tokens=max(self.max_tokens, 1600),
            )
            content = response.choices[0].message.content if response.choices else None
            if not content:
                return {
                    "success": False,
                    "model": self.model,
                    "explanations": {},
                    "error": "Groq API returned an empty response. Please click Retry to try again.",
                    "error_type": "parse_error",
                }

            parsed_json = json.loads(content)
            validated = BatchProgramAIExplanationsSchema.model_validate(parsed_json)

            explanations_by_code: Dict[str, Dict[str, Any]] = {}
            allowed_codes = {
                standardize_program_code(r.get("program_code")) or str(r.get("program_code", "")).strip()
                for r in recommendations
            }

            for item in validated.explanations:
                clean_code = standardize_program_code(item.program_code) or item.program_code.strip()
                # Guardrail: Only accept explanations for the existing shortlisted programmes
                if clean_code in allowed_codes:
                    explanations_by_code[clean_code] = {
                        "program_code": clean_code,
                        "why_this_programme": item.why_this_programme.strip(),
                        "learning_progression": item.learning_progression.strip(),
                        "professional_relevance": item.professional_relevance.strip(),
                        "suggested_pitch": item.suggested_pitch.strip(),
                        "points_to_verify": [
                            pt.strip() for pt in item.points_to_verify if pt and pt.strip()
                        ],
                    }

            # If the model returned items in order but slightly altered code formatting, map by position as fallback
            if len(explanations_by_code) < len(recommendations) and len(validated.explanations) == len(recommendations):
                for rec, item in zip(recommendations, validated.explanations):
                    c_code = standardize_program_code(rec.get("program_code")) or str(rec.get("program_code", "")).strip()
                    if c_code not in explanations_by_code:
                        explanations_by_code[c_code] = {
                            "program_code": c_code,
                            "why_this_programme": item.why_this_programme.strip(),
                            "learning_progression": item.learning_progression.strip(),
                            "professional_relevance": item.professional_relevance.strip(),
                            "suggested_pitch": item.suggested_pitch.strip(),
                            "points_to_verify": [
                                pt.strip() for pt in item.points_to_verify if pt and pt.strip()
                            ],
                        }

            if not explanations_by_code:
                return {
                    "success": False,
                    "model": self.model,
                    "explanations": {},
                    "error": "AI response did not match any of the shortlisted programme codes. Please retry.",
                    "error_type": "parse_error",
                }

            return {
                "success": True,
                "model": self.model,
                "explanations": explanations_by_code,
                "error": None,
                "error_type": None,
            }

        except json.JSONDecodeError as exc:
            return {
                "success": False,
                "model": self.model,
                "explanations": {},
                "error": f"Failed to parse AI explanation JSON response ({exc}). Please retry.",
                "error_type": "parse_error",
            }
        except Exception as exc:
            return {
                "success": False,
                "model": self.model,
                "explanations": {},
                "error": (
                    f"Unable to generate AI explanations via Groq ({type(exc).__name__}: {exc}). "
                    "Your ranked programme recommendations remain visible above. You can check your "
                    "API key/network connection and click Retry."
                ),
                "error_type": "api_error",
            }

    def generate_explanation(
        self,
        student_profile: Dict[str, Any],
        recommended_program: Dict[str, Any],
        programs_df: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """Generate structured explanation for a single recommended programme."""
        batch_res = self.generate_explanations_for_recommendations(
            student_record=student_profile,
            recommendations=[recommended_program],
            programs_df=programs_df,
        )
        if not batch_res["success"]:
            return batch_res
        prog_code = (
            standardize_program_code(recommended_program.get("program_code"))
            or str(recommended_program.get("program_code", "")).strip()
        )
        return {
            "success": True,
            "model": batch_res["model"],
            "explanation": batch_res["explanations"].get(prog_code, {}),
            "error": None,
            "error_type": None,
        }
