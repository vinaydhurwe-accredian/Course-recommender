"""Explainable Next-Course Recommendation Engine.

Independently identifies and ranks suitable next programmes by analysing the student's existing
academic and professional profile against the available programme catalogue in `programs.csv`.
"""
import re
from typing import Any, Dict, List, Optional, Set, Tuple
import pandas as pd

from config import RECOMMENDATION_CONFIG
from src.data_loader import parse_multi_value_field, standardize_program_code


STOPWORDS: Set[str] = {
    "a", "an", "and", "are", "as", "at", "batch", "be", "build", "builds", "by", "can",
    "certificate", "certification", "certified", "class", "classes", "course", "courses",
    "covers", "designed", "development", "edition", "education", "for", "from", "hours",
    "in", "includes", "including", "into", "is", "it", "its", "live", "module", "modules",
    "month", "months", "more", "new", "no", "not", "of", "offered", "on", "online", "or",
    "other", "others", "overall", "part", "past", "pillar", "pillars", "plus", "practical",
    "prepares", "program", "programme", "programmes", "programs", "real", "required",
    "school", "session", "sessions", "so", "specified", "stated", "status", "such",
    "taught", "that", "the", "their", "them", "then", "these", "they", "this", "those",
    "three", "through", "to", "tools", "total", "two", "up", "use", "using", "various",
    "want", "whether", "which", "who", "will", "with", "work", "working", "world", "year",
    "years", "accredian", "iim", "iit", "xlri", "sp", "jain", "visakhapatnam", "iimv",
    "ranchi", "lucknow", "raipur", "trichy", "jammu", "amritsar", "kashipur", "hyderabad",
    "kanpur", "delhi", "ncr", "jamshedpur",
}

SYNONYM_MAP: Dict[str, str] = {
    "genai": "generative_ai",
    "llm": "large_language_models",
    "llms": "large_language_models",
    "ml": "machine_learning",
    "dl": "deep_learning",
    "nlp": "natural_language_processing",
    "hr": "human_resources",
    "hrm": "human_resources",
    "shrm": "human_resources",
    "hrbp": "human_resources",
    "chro": "chief_human_resources_officer",
    "chros": "chief_human_resources_officer",
    "cto": "chief_technology_officer",
    "ctos": "chief_technology_officer",
    "cdaio": "chief_digital_ai_officer",
    "cdo": "chief_digital_ai_officer",
    "ceo": "chief_executive_officer",
    "ceos": "chief_executive_officer",
    "coo": "chief_operating_officer",
    "coos": "chief_operating_officer",
    "cfo": "chief_financial_officer",
    "cfos": "chief_financial_officer",
    "cxo": "cxo_executive",
    "cxos": "cxo_executive",
    "vp": "vice_president",
    "vps": "vice_president",
    "svp": "vice_president",
    "avp": "vice_president",
    "gm": "general_manager",
    "md": "managing_director",
    "pm": "project_manager",
    "pmo": "project_management",
    "bfsi": "banking_financial_services",
    "fintech": "banking_financial_services",
    "finance": "banking_financial_services",
    "financial": "banking_financial_services",
    "banking": "banking_financial_services",
    "it": "information_technology",
    "ites": "information_technology",
    "tech": "information_technology",
    "technology": "information_technology",
    "software": "information_technology",
    "pharma": "healthcare_pharma",
    "pharmaceuticals": "healthcare_pharma",
    "healthcare": "healthcare_pharma",
    "ecommerce": "retail_ecommerce",
    "retail": "retail_ecommerce",
    "fmcg": "retail_ecommerce",
    "consumer": "retail_ecommerce",
    "telecom": "telecommunications",
    "infra": "infrastructure_construction",
    "infrastructure": "infrastructure_construction",
    "construction": "infrastructure_construction",
    "consultancy": "consulting",
    "consultants": "consulting",
    "consultant": "consulting",
}

DOMAIN_CATEGORIES: Dict[str, Set[str]] = {
    "AI & Generative AI": {
        "ai", "artificial", "intelligence", "generative_ai", "agentic", "machine_learning",
        "large_language_models", "data",
    },
    "Digital & Technology Leadership": {
        "digital", "information_technology", "chief_technology_officer",
        "chief_digital_ai_officer", "architecture", "cloud", "cybersecurity", "transformation",
    },
    "Strategy & Executive Leadership": {
        "strategy", "strategic", "leadership", "executive", "cxo_executive",
        "chief_executive_officer", "general", "management", "boardroom", "governance",
    },
    "Human Resources & People Leadership": {
        "human_resources", "chief_human_resources_officer", "people", "talent",
        "workforce", "culture", "organizational",
    },
    "Project, Product & Design Innovation": {
        "project", "project_management", "product", "design", "thinking", "innovation",
        "agile", "scrum", "delivery",
    },
    "Finance & Operations Leadership": {
        "banking_financial_services", "chief_financial_officer", "chief_operating_officer",
        "operations", "supply", "chain", "logistics",
    },
}

# Complementary cross-domain progression affinity matrix (0.0 to 1.0)
CROSS_DOMAIN_COMPLEMENTARITY: Dict[Tuple[str, str], float] = {
    ("AI & Generative AI", "Digital & Technology Leadership"): 0.88,
    ("AI & Generative AI", "Strategy & Executive Leadership"): 0.85,
    ("AI & Generative AI", "Project, Product & Design Innovation"): 0.80,
    ("Digital & Technology Leadership", "AI & Generative AI"): 0.88,
    ("Digital & Technology Leadership", "Strategy & Executive Leadership"): 0.86,
    ("Digital & Technology Leadership", "Project, Product & Design Innovation"): 0.78,
    ("Strategy & Executive Leadership", "AI & Generative AI"): 0.84,
    ("Strategy & Executive Leadership", "Digital & Technology Leadership"): 0.84,
    ("Strategy & Executive Leadership", "Finance & Operations Leadership"): 0.82,
    ("Strategy & Executive Leadership", "Project, Product & Design Innovation"): 0.78,
    ("Human Resources & People Leadership", "Strategy & Executive Leadership"): 0.82,
    ("Human Resources & People Leadership", "AI & Generative AI"): 0.72,
    ("Project, Product & Design Innovation", "Strategy & Executive Leadership"): 0.85,
    ("Project, Product & Design Innovation", "AI & Generative AI"): 0.82,
    ("Project, Product & Design Innovation", "Digital & Technology Leadership"): 0.80,
    ("Finance & Operations Leadership", "Strategy & Executive Leadership"): 0.88,
    ("Finance & Operations Leadership", "AI & Generative AI"): 0.76,
    ("Finance & Operations Leadership", "Digital & Technology Leadership"): 0.76,
}


def normalize_weights(custom_weights: Optional[Dict[str, float]] = None) -> Dict[str, float]:
    """Load and normalize scoring weights so they always sum to 1.0."""
    default_weights = {
        "current_program_relevance": 0.40,
        "professional_relevance": 0.25,
        "experience_qualification_fit": 0.20,
        "skill_progression": 0.15,
    }
    cfg_weights = (RECOMMENDATION_CONFIG or {}).get("weights", {})
    merged = dict(default_weights)
    for key in default_weights:
        if custom_weights and key in custom_weights:
            merged[key] = float(custom_weights[key])
        elif key in cfg_weights:
            merged[key] = float(cfg_weights[key])

    total = sum(merged.values())
    if total <= 0:
        return default_weights
    return {k: v / total for k, v in merged.items()}


def tokenize_text(text: Any) -> Set[str]:
    """Tokenize free text into canonical normalized keywords."""
    if pd.isna(text) or text is None:
        return set()
    raw = str(text).lower().replace("e-commerce", "ecommerce").replace("ai/ml", "ai ml")
    words = re.findall(r"[a-z0-9]+", raw)
    tokens: Set[str] = set()
    for word in words:
        if len(word) < 2 or word in STOPWORDS:
            continue
        canonical = SYNONYM_MAP.get(word, word)
        tokens.add(canonical)
    return tokens


def classify_domain_themes(text: Any) -> Set[str]:
    """Map domain, title, or description text into high-level domain categories."""
    tokens = tokenize_text(text)
    matched: Set[str] = set()
    for category, keywords in DOMAIN_CATEGORIES.items():
        if tokens & keywords:
            matched.add(category)
    return matched


def infer_program_level_tier(
    program_level: Any, min_experience: Any = None, program_name: Any = None
) -> int:
    """Infer an ordinal programme seniority tier (1 = Foundational/Early, 2 = Mid-Senior, 3 = Senior/CXO)."""
    level_str = str(program_level or "").lower()
    name_str = str(program_name or "").lower()
    min_exp = float(min_experience) if pd.notna(min_experience) else None

    cxo_markers = ["chro", "cto", "cdaio", "ceo", "cfo", "coo", "cxo", "global senior"]
    if any(m in name_str for m in cxo_markers):
        return 3
    if "senior level" in level_str or "senior executive" in level_str or "senior / executive" in level_str:
        return 3
    if min_exp is not None and min_exp >= 8:
        return 3
    if "early" in level_str or (min_exp is not None and min_exp <= 3) or level_str == "intermediate":
        return 1
    return 2


def extract_student_profile(
    student_record: pd.Series, programs_df: pd.DataFrame
) -> Dict[str, Any]:
    """Step 1: Extract structured student profile and look up current programme from programs.csv."""
    student_id = str(student_record.get("student_id", "")).strip() or "Unknown"
    current_code = standardize_program_code(student_record.get("current_program_code"))

    current_program: Optional[pd.Series] = None
    if current_code and not programs_df.empty and "program_code" in programs_df.columns:
        matched = programs_df[
            programs_df["program_code"].apply(standardize_program_code) == current_code
        ]
        if not matched.empty:
            current_program = matched.iloc[0]

    exp_val = student_record.get("experience_years")
    experience_years: Optional[float] = None
    if pd.notna(exp_val):
        try:
            experience_years = float(exp_val)
        except (ValueError, TypeError):
            experience_years = None

    def _clean_opt(val: Any) -> Optional[str]:
        if pd.isna(val) or val is None:
            return None
        s = str(val).strip()
        return s if s and s.lower() not in {"nan", "none", "null", "<na>"} else None

    qualification = _clean_opt(student_record.get("qualification"))
    degree_name = _clean_opt(student_record.get("degree_name"))
    designation = _clean_opt(student_record.get("designation"))
    working_status = _clean_opt(student_record.get("working_status"))
    current_industry = _clean_opt(student_record.get("current_industry"))
    previous_industry = _clean_opt(student_record.get("previous_industry"))

    # Track data completeness coverage across key profile dimensions
    key_signals = [
        current_program is not None,
        qualification is not None or degree_name is not None,
        experience_years is not None,
        designation is not None,
        current_industry is not None or previous_industry is not None,
    ]
    coverage_ratio = sum(1 for flag in key_signals if flag) / len(key_signals)

    missing_fields: List[str] = []
    if current_program is None:
        missing_fields.append("current_program_in_catalog")
    if qualification is None:
        missing_fields.append("qualification")
    if degree_name is None:
        missing_fields.append("degree_name")
    if experience_years is None:
        missing_fields.append("experience_years")
    if designation is None:
        missing_fields.append("designation")
    if current_industry is None:
        missing_fields.append("current_industry")
    if previous_industry is None:
        missing_fields.append("previous_industry")

    return {
        "student_id": student_id,
        "current_program_code": current_code,
        "current_program": current_program,
        "qualification": qualification,
        "degree_name": degree_name,
        "experience_years": experience_years,
        "designation": designation,
        "working_status": working_status,
        "current_industry": current_industry,
        "previous_industry": previous_industry,
        "coverage_ratio": round(coverage_ratio, 2),
        "missing_fields": missing_fields,
    }


def filter_candidate_programs(
    programs_df: pd.DataFrame,
    current_program_code: Optional[str],
    preferred_domain: Optional[str] = None,
) -> pd.DataFrame:
    """Step 2: Filter candidate programmes, always excluding the student's current programme
    and optionally filtering by the user-selected domain.
    """
    if programs_df.empty:
        return programs_df.copy()

    df = programs_df.copy()
    clean_current = standardize_program_code(current_program_code)

    if clean_current and "program_code" in df.columns:
        df = df[df["program_code"].apply(standardize_program_code) != clean_current]

    if preferred_domain and preferred_domain.strip() and preferred_domain.strip() != "All Domains":
        target_dom = preferred_domain.strip()
        target_dom_lower = target_dom.lower()

        def _matches_domain(row: pd.Series) -> bool:
            raw_dom = str(row.get("domain", "")).strip()
            if raw_dom.lower() == target_dom_lower:
                return True
            dom_items = [d.strip().lower() for d in parse_multi_value_field(raw_dom)]
            return target_dom_lower in dom_items

        df = df[df.apply(_matches_domain, axis=1)]

    return df.reset_index(drop=True)


def score_current_program_relevance(
    student_profile: Dict[str, Any], candidate_row: pd.Series
) -> Tuple[float, List[str], List[str]]:
    """Calculate Current Programme Relevance score (0-100) with transparent domain, description,
    skills, and complementary cross-domain progression logic.
    """
    reasons: List[str] = []
    gaps: List[str] = []

    current_prog = student_profile.get("current_program")
    cand_name = str(candidate_row.get("program_name", candidate_row.get("program_code", ""))).strip()
    cand_domain = str(candidate_row.get("domain", "")).strip()
    cand_desc = str(candidate_row.get("program_description", "")).strip()
    cand_aud = str(candidate_row.get("target_audience", "")).strip()
    cand_skills = parse_multi_value_field(candidate_row.get("skills_covered"))

    if current_prog is None:
        # Handle missing current programme in catalogue without awarding 0:
        # Use available student degree/designation/industry signals to estimate thematic alignment
        gaps.append(
            f"Current programme '{student_profile.get('current_program_code') or 'Unknown'}' is not in programs.csv; "
            "programme-to-programme comparison used fallback profile signals (reduced data coverage)."
        )
        fallback_tokens = (
            tokenize_text(student_profile.get("degree_name"))
            | tokenize_text(student_profile.get("designation"))
            | tokenize_text(student_profile.get("current_industry"))
        )
        cand_tokens = tokenize_text(cand_domain) | tokenize_text(cand_desc) | tokenize_text(cand_name)
        if fallback_tokens and cand_tokens:
            overlap = fallback_tokens & cand_tokens
            ratio = len(overlap) / max(1, min(len(fallback_tokens), 8))
            score = min(80.0, 52.0 + ratio * 30.0)
            if overlap:
                reasons.append(
                    f"Aligns with student background keywords ({', '.join(sorted(list(overlap))[:3])}) "
                    f"in {cand_domain}."
                )
            return round(score, 1), reasons, gaps
        return 55.0, reasons, gaps

    curr_code = str(current_prog.get("program_code", "")).strip()
    curr_name = str(current_prog.get("program_name", curr_code)).strip()
    curr_domain = str(current_prog.get("domain", "")).strip()
    curr_desc = str(current_prog.get("program_description", "")).strip()
    curr_aud = str(current_prog.get("target_audience", "")).strip()
    curr_skills = parse_multi_value_field(current_prog.get("skills_covered"))

    # 1. Domain & Complementary Cross-Domain Alignment (0 to 40 pts)
    curr_themes = classify_domain_themes(f"{curr_domain} {curr_name}")
    cand_themes = classify_domain_themes(f"{cand_domain} {cand_name}")

    curr_dom_tokens = tokenize_text(curr_domain)
    cand_dom_tokens = tokenize_text(cand_domain)
    dom_jaccard = (
        len(curr_dom_tokens & cand_dom_tokens) / len(curr_dom_tokens | cand_dom_tokens)
        if (curr_dom_tokens and cand_dom_tokens)
        else 0.0
    )

    shared_themes = curr_themes & cand_themes
    best_cross_affinity = 0.0
    best_pair: Optional[Tuple[str, str]] = None
    for ct in curr_themes:
        for kt in cand_themes:
            if ct != kt:
                aff = CROSS_DOMAIN_COMPLEMENTARITY.get((ct, kt), 0.55)
                if aff > best_cross_affinity:
                    best_cross_affinity = aff
                    best_pair = (ct, kt)

    # Check if candidate is nearly an identical twin of the current programme (e.g. EPLAI at another institute)
    curr_skill_tokens = tokenize_text(" ".join(curr_skills))
    cand_skill_tokens = tokenize_text(" ".join(cand_skills))
    skill_jaccard = (
        len(curr_skill_tokens & cand_skill_tokens) / len(curr_skill_tokens | cand_skill_tokens)
        if (curr_skill_tokens and cand_skill_tokens)
        else 0.0
    )

    if shared_themes and best_cross_affinity >= 0.78 and len(cand_themes - curr_themes) > 0:
        # Ideal progression: shares core theme AND expands into a complementary theme
        domain_pts = 38.0
        new_theme = sorted(list(cand_themes - curr_themes))[0]
        reasons.append(
            f"Bridges {curr_code} ({curr_domain}) into {new_theme} ({cand_domain})."
        )
    elif shared_themes:
        if dom_jaccard >= 0.75 and skill_jaccard >= 0.60:
            # Very similar curriculum in same narrow track -> slightly lower than progressive expansion
            domain_pts = 27.0
            gaps.append(
                f"Substantial thematic overlap with {curr_code}; verify the student seeks deeper specialization in {cand_domain}."
            )
        else:
            domain_pts = 34.0 + min(4.0, dom_jaccard * 5.0)
            reasons.append(
                f"Strong domain continuity from {curr_code} ({curr_domain}) to {cand_domain}."
            )
    elif best_cross_affinity > 0:
        domain_pts = 36.0 * best_cross_affinity
        if best_pair and best_cross_affinity >= 0.75:
            reasons.append(
                f"Complementary cross-domain progression from {best_pair[0]} ({curr_code}) to {best_pair[1]} ({cand_domain})."
            )
    else:
        domain_pts = 20.0

    # 2. Programme Description & Target Audience Semantic Relevance (0 to 30 pts)
    curr_body_tokens = tokenize_text(f"{curr_desc} {curr_aud}")
    cand_body_tokens = tokenize_text(f"{cand_desc} {cand_aud}")
    if curr_body_tokens and cand_body_tokens:
        shared_body = curr_body_tokens & cand_body_tokens
        overlap_coeff = len(shared_body) / max(1, min(len(curr_body_tokens), len(cand_body_tokens)))
        # Optimal description overlap is 0.25 - 0.65 (shared business/executive context without being identical)
        if 0.20 <= overlap_coeff <= 0.70:
            desc_pts = 22.0 + ((overlap_coeff - 0.20) / 0.50) * 8.0
        elif overlap_coeff > 0.70:
            desc_pts = 24.0
        else:
            desc_pts = max(10.0, overlap_coeff * 90.0)
    else:
        desc_pts = 18.0

    # 3. Skills Bridge & Programme Level Progression (0 to 30 pts)
    curr_tier = infer_program_level_tier(
        current_prog.get("program_level"),
        current_prog.get("minimum_experience_years"),
        curr_name,
    )
    cand_tier = infer_program_level_tier(
        candidate_row.get("program_level"),
        candidate_row.get("minimum_experience_years"),
        cand_name,
    )

    if cand_tier > curr_tier:
        level_pts = 15.0
        reasons.append(
            f"Advances academic level from {current_prog.get('program_level') or 'current tier'} "
            f"to {candidate_row.get('program_level') or 'executive tier'}."
        )
    elif cand_tier == curr_tier:
        level_pts = 12.5
    else:
        level_pts = 7.0
        gaps.append(
            f"Candidate programme level ({candidate_row.get('program_level')}) is targeted at an earlier career tier than {curr_code}."
        )

    if curr_skill_tokens and cand_skill_tokens:
        if 0.12 <= skill_jaccard <= 0.55:
            skill_bridge_pts = 15.0
        elif skill_jaccard > 0.55:
            skill_bridge_pts = 10.0
        else:
            skill_bridge_pts = max(7.0, skill_jaccard * 85.0)
    else:
        # Handle missing skills list without zeroing out
        skill_bridge_pts = 11.5

    total_score = min(100.0, max(0.0, domain_pts + desc_pts + level_pts + skill_bridge_pts))
    return round(total_score, 1), reasons, gaps


def score_professional_relevance(
    student_profile: Dict[str, Any], candidate_row: pd.Series
) -> Tuple[float, List[str], List[str]]:
    """Calculate Professional Relevance score (0-100) by comparing student designation against
    target_roles and student current/previous industry against target_industries.
    """
    reasons: List[str] = []
    gaps: List[str] = []

    designation = student_profile.get("designation")
    curr_ind = student_profile.get("current_industry")
    prev_ind = student_profile.get("previous_industry")

    target_roles_raw = candidate_row.get("target_roles")
    target_roles_list = parse_multi_value_field(target_roles_raw)
    target_ind_raw = candidate_row.get("target_industries")
    target_ind_list = parse_multi_value_field(target_ind_raw)
    cand_aud_raw = str(candidate_row.get("target_audience", ""))

    # --- A. Role / Designation Matching (0 to 60 pts) ---
    cand_domain_str = str(candidate_row.get("domain", "")).lower()
    cand_name_str = str(candidate_row.get("program_name", "")).lower()
    is_hr_specialist_prog = any(
        k in f"{cand_domain_str} {cand_name_str}"
        for k in ["human resource", "chro", "digital hr", "strategic hr"]
    )
    is_finance_specialist_prog = "chief financial officer" in f"{cand_domain_str} {cand_name_str}" or candidate_row.get("program_code") == "CFO"

    if not designation:
        role_pts = 38.0
        gaps.append("Student designation is not provided; role match estimated from available profile.")
    elif not target_roles_list:
        role_pts = 44.0
    else:
        desig_lower = designation.lower()
        desig_tokens = tokenize_text(designation)
        student_is_hr = bool(desig_tokens & {"human_resources", "chief_human_resources_officer", "people", "talent", "learning"})
        student_is_finance = bool(desig_tokens & {"banking_financial_services", "chief_financial_officer", "controller", "fp", "accountant"})

        matched_role_labels: List[str] = []
        best_role_match = 0.0

        # Generic modifier tokens that should not alone trigger a match with a specialist role (e.g. "HR Director")
        generic_title_tokens = {"director", "directors", "vice_president", "manager", "managers", "head", "heads", "lead", "leads", "senior", "chief", "officer"}

        for role_item in target_roles_list:
            r_lower = role_item.lower()
            r_tokens = tokenize_text(role_item)
            role_is_hr_specific = bool(r_tokens & {"human_resources", "chief_human_resources_officer", "people", "talent", "l&d"})

            if role_is_hr_specific and not student_is_hr:
                continue

            if desig_lower in r_lower or r_lower in desig_lower:
                best_role_match = max(best_role_match, 1.0)
                matched_role_labels.append(role_item)
            elif desig_tokens and r_tokens and (desig_tokens & r_tokens):
                shared = desig_tokens & r_tokens
                # Require either a functional token match or that the target role item itself is general
                if (shared - generic_title_tokens) or not (r_tokens - generic_title_tokens):
                    overlap_ratio = len(shared) / min(len(desig_tokens), len(r_tokens))
                    score_est = min(0.95, 0.72 + overlap_ratio * 0.23)
                    if score_est > best_role_match:
                        best_role_match = score_est
                    matched_role_labels.append(role_item)

        # Seniority / functional family equivalence if no direct keyword hit
        if best_role_match < 0.75:
            if is_hr_specialist_prog and not student_is_hr:
                best_role_match = 0.40
                gaps.append(
                    f"Programme is tailored for HR/People leadership roles, whereas student is '{designation}'."
                )
            elif is_finance_specialist_prog and not student_is_finance:
                best_role_match = 0.45
                gaps.append(
                    f"Programme is tailored for Finance/CFO leadership roles, whereas student is '{designation}'."
                )
            else:
                senior_leaders = {
                    "chief", "cxo_executive", "chief_technology_officer", "chief_human_resources_officer",
                    "chief_digital_ai_officer", "chief_executive_officer", "chief_operating_officer",
                    "chief_financial_officer", "vice_president", "director", "head", "general_manager",
                    "managing_director", "president", "leader", "partner",
                }
                mid_managers = {"manager", "lead", "architect", "principal", "senior", "specialist", "analyst", "consultant"}
                all_role_tokens = tokenize_text(" ".join(target_roles_list) + " " + cand_aud_raw)

                if (desig_tokens & senior_leaders) and (all_role_tokens & senior_leaders):
                    best_role_match = max(best_role_match, 0.82)
                    if not matched_role_labels:
                        matched_role_labels.append("Senior / Executive Leadership Roles")
                elif (desig_tokens & mid_managers) and (all_role_tokens & (mid_managers | {"professionals", "career"})):
                    best_role_match = max(best_role_match, 0.76)
                    if not matched_role_labels:
                        matched_role_labels.append("Mid-Career & Managerial Roles")

        if best_role_match >= 0.72:
            role_pts = 60.0 * best_role_match
            reasons.append(
                f"Student's role ({designation}) aligns with target roles ({', '.join(matched_role_labels[:2])})."
            )
        else:
            role_pts = max(22.0, 60.0 * best_role_match)
            if not any("tailored for" in g for g in gaps):
                gaps.append(
                    f"Target roles focus on {', '.join(target_roles_list[:3])}, whereas student's current designation is '{designation}'."
                )

    # --- B. Industry Relevance (0 to 40 pts) ---
    if not curr_ind and not prev_ind:
        ind_pts = 28.0
        gaps.append("Student's current and previous industry fields are missing (reduced industry data coverage).")
    else:
        raw_ind_str = str(target_ind_raw or "").lower()
        is_cross_industry = (
            not target_ind_list
            or "not specified" in raw_ind_str
            or "cross-industry" in raw_ind_str
            or "industry-agnostic" in raw_ind_str
        )

        curr_ind_tokens = tokenize_text(curr_ind)
        prev_ind_tokens = tokenize_text(prev_ind)
        target_ind_tokens = tokenize_text(" ".join(target_ind_list) + " " + raw_ind_str)

        if curr_ind_tokens and (curr_ind_tokens & target_ind_tokens):
            ind_pts = 40.0
            reasons.append(
                f"Directly targets the student's current industry ({curr_ind})."
            )
        elif prev_ind_tokens and (prev_ind_tokens & target_ind_tokens):
            ind_pts = 35.0
            reasons.append(
                f"Connects with the student's previous industry background ({prev_ind})."
            )
        elif is_cross_industry:
            ind_pts = 34.0
            ind_label = curr_ind or prev_ind
            reasons.append(
                f"Cross-industry executive curriculum applicable to {ind_label}."
            )
        else:
            ind_pts = 22.0
            gaps.append(
                f"Programme primarily highlights {', '.join(target_ind_list[:3])}, outside student's industry ({curr_ind or prev_ind})."
            )

    total_score = min(100.0, max(0.0, role_pts + ind_pts))
    return round(total_score, 1), reasons, gaps


def score_experience_and_qualification_fit(
    student_profile: Dict[str, Any], candidate_row: pd.Series
) -> Tuple[float, List[str], List[str]]:
    """Step 3 & 4: Evaluate Experience and Qualification Fit (0-100).

    - Treats experience range as a suitability factor (does not hard-reject).
    - Does not automatically reject a student when free-text prerequisites or qualification
      descriptions (such as >50% marks or SOP) cannot be verified from CSV alone; flags them
      for counsellor verification.
    """
    reasons: List[str] = []
    gaps: List[str] = []

    exp_years = student_profile.get("experience_years")
    qualification = student_profile.get("qualification")
    degree_name = student_profile.get("degree_name")

    min_exp_raw = candidate_row.get("minimum_experience_years")
    max_exp_raw = candidate_row.get("maximum_experience_years")
    min_exp = float(min_exp_raw) if pd.notna(min_exp_raw) else None
    max_exp = float(max_exp_raw) if pd.notna(max_exp_raw) else None

    qual_req_raw = str(candidate_row.get("qualification_requirements", "") or "").strip()
    prereq_raw = str(candidate_row.get("prerequisites", "") or "").strip()
    cand_level = str(candidate_row.get("program_level", "") or "").strip()

    # If min_exp is missing from numeric column, check if mentioned in prerequisites text
    if min_exp is None and prereq_raw:
        m = re.search(r"(\d+)\s*\+?\s*years?\s+of\s+(?:total\s+)?(?:work\s+|professional\s+)?experience", prereq_raw, re.I)
        if m:
            min_exp = float(m.group(1))

    # --- A. Experience Suitability (0 to 65 pts) ---
    if exp_years is None:
        exp_pts = 42.0
        gaps.append(
            "Counsellor Verification: Student's work experience years are missing; verify experience suitability manually."
        )
    elif min_exp is None and max_exp is None:
        # Infer suitability from programme level
        tier = infer_program_level_tier(cand_level, None, candidate_row.get("program_name"))
        if tier == 3 and exp_years >= 10:
            exp_pts = 60.0
            reasons.append(
                f"{exp_years:.0f} years of experience suits this {cand_level or 'senior executive'} programme."
            )
        elif tier == 3 and exp_years < 8:
            exp_pts = 38.0
            gaps.append(
                f"Suitability Note: {cand_level or 'Senior executive'} cohort typically has 10+ years of experience (student has {exp_years:.0f} yrs)."
            )
        else:
            exp_pts = 56.0
    else:
        eff_min = min_exp if min_exp is not None else 0.0
        if exp_years < eff_min:
            deficit = eff_min - exp_years
            # Proportional suitability penalty rather than hard rejection
            exp_pts = max(15.0, 55.0 - deficit * 8.5)
            gaps.append(
                f"Counsellor Verification (Experience): Programme suggests {eff_min:.0f}+ years of work experience, "
                f"while student has {exp_years:.0f} year(s)."
            )
        elif max_exp is not None and exp_years > max_exp:
            excess = exp_years - max_exp
            exp_pts = max(30.0, 58.0 - excess * 4.0)
            gaps.append(
                f"Student's {exp_years:.0f} years of experience exceeds the suggested maximum of {max_exp:.0f} years."
            )
        else:
            # Student meets minimum experience. Check how closely the programme's cohort band fits the student's seniority.
            over_min = exp_years - eff_min
            if over_min <= 7.0:
                exp_pts = 65.0
                reasons.append(
                    f"Student's {exp_years:.0f} years of experience fits the {eff_min:.0f}+ years requirement well."
                )
            elif over_min <= 12.0:
                exp_pts = 60.0
                reasons.append(
                    f"Meets the {eff_min:.0f}+ years experience requirement ({exp_years:.0f} years total)."
                )
            else:
                # Highly experienced leader considering a low-minimum (e.g. 1+ or 3+ yr) programme
                exp_pts = 48.0
                if eff_min <= 3.0:
                    gaps.append(
                        f"Programme requires only {eff_min:.0f}+ years of experience; with {exp_years:.0f} years, the student may prefer a more senior cohort."
                    )
                else:
                    reasons.append(
                        f"Exceeds the {eff_min:.0f}+ years experience requirement ({exp_years:.0f} years total)."
                    )

    # --- B. Qualification & Prerequisite Verification (0 to 35 pts) ---
    recognized_grad_quals = {
        "mba", "pgdm", "m.tech", "mtech", "m.sc", "msc", "mca", "ma", "phd", "ph.d",
        "b.tech", "btech", "b.e.", "be", "b.com", "bcom", "ca", "bba", "b.sc", "bsc", "llb",
    }
    qual_clean = (qualification or "").lower().strip()
    has_graduate_degree = any(q in qual_clean for q in recognized_grad_quals) or bool(degree_name)

    if has_graduate_degree:
        qual_pts = 35.0
        deg_display = degree_name or qualification
        reasons.append(
            f"Holds recognized qualification ({deg_display}) matching graduate eligibility."
        )
    else:
        qual_pts = 22.0
        gaps.append(
            "Counsellor Verification (Qualification): Student qualification/degree details are incomplete; verify graduate degree eligibility."
        )

    # Flag un-verifiable free-text conditions in qualification_requirements / prerequisites for counsellor review
    combined_req_text = f"{qual_req_raw} ; {prereq_raw}".lower()
    verification_flags: List[str] = []
    if "50%" in combined_req_text or "45%" in combined_req_text:
        verification_flags.append("minimum 50% aggregate marks in graduation")
    if "sop" in combined_req_text:
        verification_flags.append("Statement of Purpose (SOP) & profile screening")
    if "mast" in combined_req_text or "personal interview" in combined_req_text:
        verification_flags.append("admission test (MAST) & personal interview")

    if verification_flags:
        gaps.append(
            f"Counsellor Verification (Prerequisites): Confirm {', '.join(verification_flags)}."
        )

    total_score = min(100.0, max(0.0, exp_pts + qual_pts))
    return round(total_score, 1), reasons, gaps


def score_skill_progression(
    student_profile: Dict[str, Any], candidate_row: pd.Series
) -> Tuple[float, List[str], List[str]]:
    """Calculate Skill Progression score (0-100) by evaluating how the candidate programme
    builds on the student's existing learning background and introduces high-value new skills.
    """
    reasons: List[str] = []
    gaps: List[str] = []

    current_prog = student_profile.get("current_program")
    cand_skills_list = parse_multi_value_field(candidate_row.get("skills_covered"))

    if not cand_skills_list:
        gaps.append("Candidate programme has no structured skills_covered list in catalogue (estimated from description).")
        cand_desc_tokens = tokenize_text(candidate_row.get("program_description"))
        if cand_desc_tokens:
            return 65.0, reasons, gaps
        return 55.0, reasons, gaps

    # Build student's existing skill/knowledge baseline from current programme + degree + role
    curr_skills_list: List[str] = []
    if current_prog is not None:
        curr_skills_list = parse_multi_value_field(current_prog.get("skills_covered"))

    curr_skill_tokens = tokenize_text(" ".join(curr_skills_list))
    degree_role_tokens = (
        tokenize_text(student_profile.get("degree_name"))
        | tokenize_text(student_profile.get("designation"))
    )
    baseline_tokens = curr_skill_tokens | degree_role_tokens

    # Classify each candidate skill phrase as either reinforcing existing foundations or expanding new capabilities
    reinforcing_skills: List[str] = []
    new_expansion_skills: List[str] = []

    for skill_phrase in cand_skills_list:
        s_tokens = tokenize_text(skill_phrase)
        if not s_tokens:
            continue
        if baseline_tokens and len(s_tokens & baseline_tokens) / len(s_tokens) >= 0.35:
            reinforcing_skills.append(skill_phrase)
        else:
            new_expansion_skills.append(skill_phrase)

    total_cand_skills = max(1, len(reinforcing_skills) + len(new_expansion_skills))
    new_ratio = len(new_expansion_skills) / total_cand_skills
    shared_ratio = len(reinforcing_skills) / total_cand_skills

    # Ideal skill progression: 15% - 50% foundational bridge + 50% - 85% new capability expansion
    if 0.50 <= new_ratio <= 0.88 and len(reinforcing_skills) >= 1:
        progression_pts = 88.0 + min(12.0, shared_ratio * 25.0)
    elif new_ratio > 0.88:
        # Almost entirely new skills -> strong expansion, slightly smaller bridge
        progression_pts = 78.0
    elif 0.30 <= new_ratio < 0.50:
        progression_pts = 72.0
    else:
        # >70% of skills already covered in current programme -> redundant
        progression_pts = 48.0
        gaps.append(
            "High curriculum skill overlap with student's current programme; limited new skill expansion."
        )

    # Bonus if new expansion skills align with the student's professional designation or target seniority
    role_tokens = tokenize_text(student_profile.get("designation"))
    new_skill_tokens = tokenize_text(" ".join(new_expansion_skills))
    if role_tokens and (role_tokens & new_skill_tokens):
        progression_pts = min(100.0, progression_pts + 5.0)

    if new_expansion_skills:
        preview_new = ", ".join(new_expansion_skills[:3])
        if reinforcing_skills:
            reasons.append(
                f"Builds on {reinforcing_skills[0]} while adding new capabilities in {preview_new}."
            )
        else:
            reasons.append(
                f"Expands skill portfolio with {preview_new}."
            )

    return round(min(100.0, max(0.0, progression_pts)), 1), reasons, gaps


def score_candidate_program(
    student_profile: Dict[str, Any],
    candidate_row: pd.Series,
    weights: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Step 4: Score a single candidate programme across all four weighted dimensions."""
    norm_weights = normalize_weights(weights)

    c_score, c_reasons, c_gaps = score_current_program_relevance(student_profile, candidate_row)
    p_score, p_reasons, p_gaps = score_professional_relevance(student_profile, candidate_row)
    e_score, e_reasons, e_gaps = score_experience_and_qualification_fit(student_profile, candidate_row)
    s_score, s_reasons, s_gaps = score_skill_progression(student_profile, candidate_row)

    overall_score = (
        c_score * norm_weights["current_program_relevance"]
        + p_score * norm_weights["professional_relevance"]
        + e_score * norm_weights["experience_qualification_fit"]
        + s_score * norm_weights["skill_progression"]
    )

    # Deduplicate reasons and potential gaps while preserving order
    all_reasons: List[str] = []
    for r in c_reasons + p_reasons + e_reasons + s_reasons:
        if r and r not in all_reasons:
            all_reasons.append(r)

    all_gaps: List[str] = []
    for g in c_gaps + p_gaps + e_gaps + s_gaps:
        if g and g not in all_gaps:
            all_gaps.append(g)

    prereq_val = candidate_row.get("prerequisites")
    if pd.isna(prereq_val) or not str(prereq_val).strip():
        qual_fallback = candidate_row.get("qualification_requirements")
        prerequisites_text = (
            str(qual_fallback).strip()
            if pd.notna(qual_fallback) and str(qual_fallback).strip()
            else "No specific prerequisites stated in catalogue."
        )
    else:
        prerequisites_text = str(prereq_val).strip()

    return {
        "program_code": str(candidate_row.get("program_code", "")).strip(),
        "program_name": str(candidate_row.get("program_name", "")).strip(),
        "domain": str(candidate_row.get("domain", "")).strip(),
        "program_level": str(candidate_row.get("program_level", "") or "Not Specified").strip(),
        "duration": str(candidate_row.get("duration", "") or "Not Specified").strip(),
        "minimum_experience_years": (
            float(candidate_row.get("minimum_experience_years"))
            if pd.notna(candidate_row.get("minimum_experience_years"))
            else None
        ),
        "maximum_experience_years": (
            float(candidate_row.get("maximum_experience_years"))
            if pd.notna(candidate_row.get("maximum_experience_years"))
            else None
        ),
        "qualification_requirements": (
            str(candidate_row.get("qualification_requirements")).strip()
            if pd.notna(candidate_row.get("qualification_requirements"))
            else "Not Specified"
        ),
        "overall_match_score": round(overall_score, 1),
        "current_program_relevance_score": round(c_score, 1),
        "professional_relevance_score": round(p_score, 1),
        "experience_qualification_score": round(e_score, 1),
        "skill_progression_score": round(s_score, 1),
        "match_reasons": all_reasons,
        "potential_gaps": all_gaps,
        "prerequisites": prerequisites_text,
        "data_coverage_ratio": student_profile.get("coverage_ratio", 1.0),
    }


def rank_candidate_programs(
    student_record: pd.Series,
    programs_df: pd.DataFrame,
    top_k: int = 3,
    preferred_domain: Optional[str] = None,
    weights: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Step 5 & 6: Score, rank, and return the top-K recommended programmes along with
    data coverage warnings and student profile metadata.
    """
    student_profile = extract_student_profile(student_record, programs_df)
    candidates_df = filter_candidate_programs(
        programs_df=programs_df,
        current_program_code=student_profile.get("current_program_code"),
        preferred_domain=preferred_domain,
    )

    warnings_list: List[str] = []
    if student_profile.get("current_program") is None:
        curr_code = student_profile.get("current_program_code") or "MISSING"
        warnings_list.append(
            f"Current programme '{curr_code}' was not found in programs.csv. "
            "Recommendations are computed from the student's academic and professional profile with reduced data coverage."
        )

    if student_profile.get("coverage_ratio", 1.0) < 0.8:
        missing_str = ", ".join(student_profile.get("missing_fields", []))
        warnings_list.append(
            f"Reduced profile data coverage ({int(student_profile['coverage_ratio'] * 100)}% available; "
            f"missing: {missing_str}). Scores are calculated from available evidence and should be verified by a counsellor."
        )

    if candidates_df.empty:
        warnings_list.append(
            "No candidate programmes matched the selected domain filter after excluding the student's current programme."
        )
        return {
            "student_id": student_profile["student_id"],
            "current_program_code": student_profile["current_program_code"],
            "selected_domain": preferred_domain or "All Domains",
            "data_coverage_ratio": student_profile["coverage_ratio"],
            "warnings": warnings_list,
            "recommendations": [],
        }

    scored_candidates: List[Dict[str, Any]] = []
    for _, cand_row in candidates_df.iterrows():
        scored = score_candidate_program(
            student_profile=student_profile,
            candidate_row=cand_row,
            weights=weights,
        )
        scored_candidates.append(scored)

    # Sort descending by overall_match_score, then current_program_relevance_score, then professional_relevance_score
    scored_candidates.sort(
        key=lambda item: (
            item["overall_match_score"],
            item["current_program_relevance_score"],
            item["professional_relevance_score"],
            item["skill_progression_score"],
        ),
        reverse=True,
    )

    top_results = scored_candidates[: max(1, int(top_k))]
    for rank_idx, rec in enumerate(top_results, start=1):
        rec["rank"] = rank_idx

    return {
        "student_id": student_profile["student_id"],
        "current_program_code": student_profile["current_program_code"],
        "selected_domain": preferred_domain or "All Domains",
        "data_coverage_ratio": student_profile["coverage_ratio"],
        "weights_used": normalize_weights(weights),
        "warnings": warnings_list,
        "recommendations": top_results,
    }


class CourseRecommender:
    """Core recommendation scoring engine."""

    def __init__(self, programs_df: pd.DataFrame, config: Optional[Dict[str, Any]] = None):
        self.programs_df = programs_df
        self.config = config or RECOMMENDATION_CONFIG
        self.weights = normalize_weights((self.config or {}).get("weights"))

    def recommend_for_student(
        self,
        student_record: pd.Series,
        top_k: Optional[int] = None,
        preferred_domain: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Return the top-K ranked programme recommendations for a student."""
        k = top_k if top_k is not None else int(self.config.get("top_k", 3))
        report = rank_candidate_programs(
            student_record=student_record,
            programs_df=self.programs_df,
            top_k=k,
            preferred_domain=preferred_domain,
            weights=self.weights,
        )
        return report["recommendations"]

    def evaluate_student_recommendations(
        self,
        student_record: pd.Series,
        top_k: Optional[int] = None,
        preferred_domain: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Return the full recommendation evaluation payload (including coverage warnings and top-K list)."""
        k = top_k if top_k is not None else int(self.config.get("top_k", 3))
        return rank_candidate_programs(
            student_record=student_record,
            programs_df=self.programs_df,
            top_k=k,
            preferred_domain=preferred_domain,
            weights=self.weights,
        )
