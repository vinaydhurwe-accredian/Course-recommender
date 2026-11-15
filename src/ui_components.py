"""Reusable UI components for the Student Next-Course Recommendation System."""
import html
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from src.ai_explainer import build_deterministic_explanation_for_program
from src.data_loader import standardize_program_code


def load_custom_css(css_path: Path) -> None:
    """Inject custom CSS rules from assets/styles.css into the Streamlit page."""
    if css_path.exists():
        css_content = css_path.read_text(encoding="utf-8")
        st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)


def format_display_value(value: Any, fallback: str = "Not Provided", suffix: str = "") -> str:
    """Format a field value cleanly for UI display, handling NaN/None gracefully."""
    if pd.isna(value) or value is None:
        return fallback
    if isinstance(value, float):
        if value.is_integer():
            text = str(int(value))
        else:
            text = f"{value:.1f}"
    else:
        text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null", "<na>"}:
        return fallback
    return f"{text}{suffix}"


def format_status_label(status: Any) -> str:
    """Convert canonical enrollment_status codes into human-readable labels."""
    if pd.isna(status) or not status:
        return "Unknown"
    mapping = {
        "completed": "Completed",
        "in_progress": "In Progress",
        "withdrawn": "Withdrawn",
    }
    raw = str(status).strip().lower()
    return mapping.get(raw, raw.replace("_", " ").title())


def get_unique_domains_from_programs(programs_df: pd.DataFrame) -> List[str]:
    """Extract sorted unique domain options dynamically from programs.csv."""
    if programs_df.empty or "domain" not in programs_df.columns:
        return ["All Domains"]

    unique_domains = sorted(
        programs_df["domain"]
        .dropna()
        .astype(str)
        .str.strip()
        .loc[lambda s: s != ""]
        .unique()
        .tolist()
    )
    return ["All Domains"] + unique_domains


def render_app_header(title: str = "Student Next-Course Recommendation System") -> None:
    """Render the top SaaS application header bar."""
    safe_title = html.escape(title)
    st.markdown(
        f"""
        <div class="saas-header">
            <div class="saas-header-left">
                <div class="saas-brand-badge">A</div>
                <div>
                    <h1 class="saas-title">{safe_title}</h1>
                    <p class="saas-subtitle">Internal Academic Counselling &amp; Programme Progression Workspace</p>
                </div>
            </div>
            <div>
                <span class="saas-pill">Counselling &amp; Sales Portal</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_summary_metric_cards(
    total_students: int,
    nearing_completion_count: int,
    completed_count: int,
    available_programmes_count: int,
    min_completion_threshold: float = 80.0,
) -> None:
    """Render the four summary metric cards on the Dashboard screen."""
    col1, col2, col3, col4 = st.columns(4, gap="medium")

    cards = [
        (
            col1,
            "Total Students",
            f"{total_students:,}",
            "Eligible in recommendation queue",
            "dot-red",
            True,
        ),
        (
            col2,
            "Students Nearing Completion",
            f"{nearing_completion_count:,}",
            f"In progress (\u2265 {min_completion_threshold:.0f}% completed)",
            "dot-amber",
            False,
        ),
        (
            col3,
            "Completed Students",
            f"{completed_count:,}",
            "100% current programme completed",
            "dot-green",
            False,
        ),
        (
            col4,
            "Available Programmes",
            f"{available_programmes_count:,}",
            "Active executive catalogue",
            "dot-slate",
            False,
        ),
    ]

    for col, label, value, subtext, dot_class, is_accent in cards:
        accent_class = "metric-card-accent" if is_accent else ""
        with col:
            st.markdown(
                f"""
                <div class="metric-card {accent_class}">
                    <div class="metric-label">{html.escape(label)}</div>
                    <div class="metric-value">{html.escape(value)}</div>
                    <div class="metric-subtext">
                        <span class="metric-dot {dot_class}"></span>
                        <span>{html.escape(subtext)}</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_student_filters(eligible_students_df: pd.DataFrame) -> pd.DataFrame:
    """Render search and filter controls for the eligible student queue and return the filtered DataFrame."""
    st.markdown(
        """
        <div class="section-header" style="margin-top: 0.75rem; margin-bottom: 0.75rem;">
            <div class="section-title">
                <span class="section-title-accent"></span>
                <span>Recommendation Queue &amp; Student Directory</span>
            </div>
            <span class="section-badge">Eligible Candidates Only</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    f_col1, f_col2, f_col3, f_col4 = st.columns([1.4, 1.3, 1.2, 1.1], gap="small")

    program_options = ["All Programmes"] + sorted(
        eligible_students_df["current_program_code"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    industry_options = ["All Industries"] + sorted(
        eligible_students_df["current_industry"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    status_raw_values = (
        eligible_students_df["enrollment_status"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )
    status_display_map = {"All Statuses": None}
    for s in sorted(status_raw_values):
        status_display_map[format_status_label(s)] = s

    with f_col1:
        student_id_query = st.text_input(
            "Search Student ID",
            placeholder="e.g. STU001, STU031...",
            key="dashboard_search_student_id",
        )
    with f_col2:
        selected_program = st.selectbox(
            "Current Programme",
            options=program_options,
            index=0,
            key="dashboard_filter_program",
        )
    with f_col3:
        selected_industry = st.selectbox(
            "Industry",
            options=industry_options,
            index=0,
            key="dashboard_filter_industry",
        )
    with f_col4:
        selected_status_label = st.selectbox(
            "Enrollment Status",
            options=list(status_display_map.keys()),
            index=0,
            key="dashboard_filter_status",
        )

    filtered_df = eligible_students_df.copy()

    if student_id_query and student_id_query.strip():
        q = student_id_query.strip().lower()
        filtered_df = filtered_df[
            filtered_df["student_id"].astype(str).str.lower().str.contains(q, na=False)
        ]

    if selected_program != "All Programmes":
        filtered_df = filtered_df[
            filtered_df["current_program_code"] == selected_program
        ]

    if selected_industry != "All Industries":
        filtered_df = filtered_df[
            filtered_df["current_industry"] == selected_industry
        ]

    target_status = status_display_map.get(selected_status_label)
    if target_status is not None:
        filtered_df = filtered_df[
            filtered_df["enrollment_status"] == target_status
        ]

    return filtered_df.reset_index(drop=True)


def render_student_table(
    filtered_df: pd.DataFrame,
    total_eligible_count: int,
    on_open_student: Optional[Callable[[str], None]] = None,
) -> None:
    """Render the formatted student recommendation queue table with quick-select navigation."""
    caption_col, action_col = st.columns([3, 2], gap="small")
    with caption_col:
        st.caption(
            f"Showing **{len(filtered_df)}** of **{total_eligible_count}** eligible students "
            "(Click a row in the table or use the quick selector to open a student profile)."
        )

    if filtered_df.empty:
        st.info("No eligible students match the selected search and filter criteria.")
        return

    display_df = pd.DataFrame(
        {
            "Student ID": filtered_df["student_id"].apply(
                lambda x: format_display_value(x, fallback="—")
            ),
            "Current Programme": filtered_df["current_program_code"].apply(
                lambda x: format_display_value(x, fallback="—")
            ),
            "Enrollment Status": filtered_df["enrollment_status"].apply(format_status_label),
            "Completion Percentage": filtered_df["completion_percentage"].apply(
                lambda x: float(x) if pd.notna(x) else 0.0
            ),
            "Designation": filtered_df["designation"].apply(
                lambda x: format_display_value(x, fallback="Not Specified")
            ),
            "Industry": filtered_df["current_industry"].apply(
                lambda x: format_display_value(x, fallback="Not Specified")
            ),
            "Experience": filtered_df["experience_years"].apply(
                lambda x: f"{int(x)} yrs" if pd.notna(x) else "Not Specified"
            ),
        }
    )

    event = st.dataframe(
        display_df,
        width="stretch",
        hide_index=True,
        height=420,
        on_select="rerun",
        selection_mode="single-row",
        column_config={
            "Student ID": st.column_config.TextColumn("Student ID", width="small"),
            "Current Programme": st.column_config.TextColumn("Current Programme", width="medium"),
            "Enrollment Status": st.column_config.TextColumn("Enrollment Status", width="small"),
            "Completion Percentage": st.column_config.ProgressColumn(
                "Completion Percentage",
                format="%.0f%%",
                min_value=0,
                max_value=100,
                width="small",
            ),
            "Designation": st.column_config.TextColumn("Designation", width="medium"),
            "Industry": st.column_config.TextColumn("Industry", width="small"),
            "Experience": st.column_config.TextColumn("Experience", width="small"),
        },
    )

    # If the user selects a row directly in the table, provide a prompt/action to open that student
    if event and hasattr(event, "selection") and event.selection.rows:
        selected_row_idx = event.selection.rows[0]
        if 0 <= selected_row_idx < len(filtered_df):
            picked_student_id = str(filtered_df.iloc[selected_row_idx]["student_id"])
            if on_open_student is not None:
                bar_col1, bar_col2 = st.columns([3.5, 1.5], gap="small")
                with bar_col1:
                    st.markdown(
                        f"**Selected Candidate:** `{picked_student_id}` — "
                        f"{format_display_value(filtered_df.iloc[selected_row_idx].get('designation'))} "
                        f"({format_display_value(filtered_df.iloc[selected_row_idx].get('current_program_code'))})"
                    )
                with bar_col2:
                    if st.button(
                        f"View {picked_student_id} Profile & Recommend \u2192",
                        type="primary",
                        width="stretch",
                        key=f"open_selected_row_{picked_student_id}",
                    ):
                        on_open_student(picked_student_id)


def render_student_profile_cards(
    student_row: pd.Series,
    program_record: Optional[pd.Series] = None,
    missing_program_warning: Optional[str] = None,
) -> None:
    """Render the selected student's banner, Academic Information card, and Professional Information card."""
    student_id = format_display_value(student_row.get("student_id"), fallback="Unknown ID")
    current_code = format_display_value(student_row.get("current_program_code"), fallback="Not Assigned")

    if program_record is not None and pd.notna(program_record.get("program_name")):
        prog_name = str(program_record.get("program_name")).strip()
        current_programme_display = f"{current_code} — {prog_name}"
    else:
        current_programme_display = current_code

    status_raw = str(student_row.get("enrollment_status", "")).strip().lower()
    status_label = format_status_label(status_raw)
    comp_pct = format_display_value(student_row.get("completion_percentage"), fallback="0", suffix="%")

    badge_class = "status-completed" if status_raw == "completed" else "status-in-progress"
    designation = format_display_value(student_row.get("designation"), fallback="Designation Not Provided")
    current_ind = format_display_value(student_row.get("current_industry"), fallback="Industry Not Provided")

    st.markdown(
        f"""
        <div class="profile-banner">
            <div>
                <h2 class="profile-id">{html.escape(student_id)}</h2>
                <p class="profile-role">{html.escape(designation)} &bull; {html.escape(current_ind)}</p>
            </div>
            <div style="display: flex; align-items: center; gap: 0.75rem;">
                <span class="status-badge {badge_class}">
                    {html.escape(status_label)} ({html.escape(comp_pct)})
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if missing_program_warning:
        st.warning(missing_program_warning)

    col_academic, col_professional = st.columns(2, gap="medium")

    # Academic information fields
    qualification = format_display_value(student_row.get("qualification"))
    degree_name = format_display_value(student_row.get("degree_name"))
    completion_year = format_display_value(student_row.get("completion_year"))

    with col_academic:
        st.markdown(
            f"""
            <div class="section-card">
                <div class="section-header">
                    <div class="section-title">
                        <span class="section-title-accent"></span>
                        <span>Academic Information</span>
                    </div>
                    <span class="section-badge">Education &amp; Enrolment</span>
                </div>
                <div class="info-grid">
                    <div class="info-item info-item-full">
                        <div class="info-label">Current Programme</div>
                        <div class="info-value">{html.escape(current_programme_display)}</div>
                    </div>
                    <div class="info-item">
                        <div class="info-label">Qualification</div>
                        <div class="info-value">{html.escape(qualification)}</div>
                    </div>
                    <div class="info-item">
                        <div class="info-label">Degree Completion Year</div>
                        <div class="info-value">{html.escape(completion_year)}</div>
                    </div>
                    <div class="info-item info-item-full">
                        <div class="info-label">Degree Name</div>
                        <div class="info-value">{html.escape(degree_name)}</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Professional information fields
    experience = format_display_value(student_row.get("experience_years"), suffix=" years")
    previous_ind = format_display_value(student_row.get("previous_industry"))
    working_status_raw = format_display_value(student_row.get("working_status"))
    if working_status_raw.lower() == "yes":
        working_status = "Currently Working (Yes)"
    elif working_status_raw.lower() == "no":
        working_status = "Not Working (No)"
    else:
        working_status = working_status_raw

    with col_professional:
        st.markdown(
            f"""
            <div class="section-card">
                <div class="section-header">
                    <div class="section-title">
                        <span class="section-title-accent"></span>
                        <span>Professional Information</span>
                    </div>
                    <span class="section-badge">Career Background</span>
                </div>
                <div class="info-grid">
                    <div class="info-item">
                        <div class="info-label">Designation</div>
                        <div class="info-value">{html.escape(designation)}</div>
                    </div>
                    <div class="info-item">
                        <div class="info-label">Experience</div>
                        <div class="info-value">{html.escape(experience)}</div>
                    </div>
                    <div class="info-item">
                        <div class="info-label">Current Industry</div>
                        <div class="info-value">{html.escape(current_ind)}</div>
                    </div>
                    <div class="info-item">
                        <div class="info-label">Previous Industry</div>
                        <div class="info-value">{html.escape(previous_ind)}</div>
                    </div>
                    <div class="info-item info-item-full">
                        <div class="info-label">Working Status</div>
                        <div class="info-value">{html.escape(working_status)}</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _copy_text_to_clipboard_js(text: str) -> None:
    """Trigger browser clipboard copy for the given pitch text."""
    payload = json.dumps(str(text))
    components.html(
        f"""
        <script>
        (function() {{
            const text = {payload};
            if (navigator.clipboard && window.isSecureContext) {{
                navigator.clipboard.writeText(text).catch(function() {{}});
            }} else {{
                const ta = document.createElement('textarea');
                ta.value = text;
                ta.style.position = 'fixed';
                ta.style.opacity = '0';
                document.body.appendChild(ta);
                ta.select();
                try {{ document.execCommand('copy'); }} catch (e) {{}}
                document.body.removeChild(ta);
            }}
        }})();
        </script>
        """,
        height=0,
    )


def render_recommendation_results(
    evaluation_payload: Dict[str, Any],
    student_row: Optional[pd.Series] = None,
    programs_df: Optional[pd.DataFrame] = None,
    ai_result: Optional[Dict[str, Any]] = None,
    on_generate_again: Optional[Callable[[], None]] = None,
    on_retry_ai: Optional[Callable[[], None]] = None,
) -> None:
    """Render unified top-3 recommendation cards with relevance scores out of 100,
    highest-ranked highlight, 5-part counselling rationale/pitch, 'Copy Pitch' button,
    and 'Generate Again' button.
    """
    student_id = html.escape(str(evaluation_payload.get("student_id", "")))
    selected_domain = html.escape(str(evaluation_payload.get("selected_domain", "All Domains")))
    coverage_ratio = float(evaluation_payload.get("data_coverage_ratio", 1.0))
    warnings_list: List[str] = evaluation_payload.get("warnings", [])
    recommendations: List[Dict[str, Any]] = evaluation_payload.get("recommendations", [])

    ai_success = bool(ai_result and ai_result.get("success"))
    ai_explanations_map: Dict[str, Dict[str, Any]] = (
        ai_result.get("explanations", {}) if ai_success and ai_result else {}
    )
    ai_model_used = (
        str(ai_result.get("model", "Groq")) if ai_result else "Engine Structured Rationale"
    )

    # Header bar with "Generate Again" button
    hdr_col1, hdr_col2 = st.columns([3.5, 1.5], gap="medium")
    with hdr_col1:
        source_badge = (
            f"GenAI Active ({html.escape(ai_model_used)})"
            if ai_success
            else "Catalogue-Grounded Rationale"
        )
        st.markdown(
            f"""
            <div class="section-header" style="margin-top: 1.25rem; margin-bottom: 0.5rem;">
                <div class="section-title">
                    <span class="section-title-accent"></span>
                    <span>Top Recommended Next Programmes &mdash; {student_id}</span>
                </div>
                <div style="display: flex; align-items: center; gap: 0.5rem; flex-wrap: wrap;">
                    <span class="section-badge">Data Coverage: {int(coverage_ratio * 100)}%</span>
                    <span class="ai-model-pill">{source_badge}</span>
                    <span class="saas-pill">Domain: {selected_domain}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with hdr_col2:
        st.markdown("<div style='height: 1.15rem;'></div>", unsafe_allow_html=True)
        if on_generate_again is not None:
            if st.button(
                "\u21bb Generate Again",
                type="secondary",
                width="stretch",
                key="generate_again_recommendations_btn",
            ):
                on_generate_again()

    for warn_msg in warnings_list:
        st.warning(warn_msg)

    # If Groq API encountered an error or missing key, keep recommendations visible, show helpful notice, and allow retry
    if ai_result and not ai_result.get("success"):
        err_type = ai_result.get("error_type")
        err_text = ai_result.get("error") or "Groq API call did not succeed."
        if err_type == "missing_key":
            st.info(
                f"ℹ️ **GenAI Status:** {err_text} "
                "Each recommendation card below includes full deterministic explanations and pitches derived from the catalogue."
            )
        else:
            err_col1, err_col2 = st.columns([3.6, 1.4], gap="small")
            with err_col1:
                st.warning(f"**GenAI Notice:** {err_text}")
            with err_col2:
                if on_retry_ai is not None:
                    if st.button(
                        "\u21bb Retry AI Explanations",
                        type="secondary",
                        width="stretch",
                        key="retry_ai_only_btn",
                    ):
                        on_retry_ai()

    if not recommendations:
        st.info("No suitable candidate programmes found for the selected filter criteria.")
        return

    st.caption(
        "Scores represent **programme relevance out of 100** based on weighted alignment "
        "(Current Programme Relevance 40%, Professional Relevance 25%, Experience & Qualification Fit 20%, "
        "Skill Progression 15%) — not probabilities of enrolment or career outcome."
    )

    for idx, rec in enumerate(recommendations, start=1):
        rank = int(rec.get("rank", idx))
        is_top_pick = rank == 1

        raw_code = str(rec.get("program_code", "")).strip()
        clean_code = standardize_program_code(raw_code) or raw_code
        prog_code = html.escape(raw_code)
        prog_name = html.escape(str(rec.get("program_name", "")))
        domain = html.escape(str(rec.get("domain", "")))
        duration = html.escape(str(rec.get("duration", "Not Specified")))
        level = html.escape(str(rec.get("program_level", "Not Specified")))

        overall = float(rec.get("overall_match_score", 0.0))
        c_score = float(rec.get("current_program_relevance_score", 0.0))
        p_score = float(rec.get("professional_relevance_score", 0.0))
        e_score = float(rec.get("experience_qualification_score", 0.0))
        s_score = float(rec.get("skill_progression_score", 0.0))

        # Resolve 5-part explanation & pitch from GenAI (if available) or deterministic catalogue fallback
        exp_data = ai_explanations_map.get(clean_code) or ai_explanations_map.get(raw_code)
        if not exp_data:
            exp_data = build_deterministic_explanation_for_program(
                student_record=student_row if student_row is not None else {},
                rec=rec,
                programs_df=programs_df,
            )

        why_matches_text = html.escape(str(exp_data.get("why_this_programme", "")))
        builds_on_course_text = html.escape(str(exp_data.get("learning_progression", "")))
        prof_relevance_text = html.escape(str(exp_data.get("professional_relevance", "")))
        raw_pitch_text = str(exp_data.get("suggested_pitch", "")).strip()
        safe_pitch_text = html.escape(raw_pitch_text)

        verify_points: List[str] = list(exp_data.get("points_to_verify", []))
        prereqs_raw = str(rec.get("prerequisites", "No specific prerequisites stated.")).strip()
        if prereqs_raw and not any(prereqs_raw[:35].lower() in pt.lower() for pt in verify_points):
            verify_points.insert(0, f"Catalogue Prerequisites: {prereqs_raw}")

        verify_html = "".join(
            f"<li style='margin-bottom: 0.3rem; color: #92400E;'>{html.escape(str(pt))}</li>"
            for pt in verify_points
        ) or f"<li style='color: #92400E;'>{html.escape(prereqs_raw)}</li>"

        card_class = "rec-card rec-card-top-pick" if is_top_pick else "rec-card"
        if is_top_pick:
            rank_badges_html = (
                '<span class="rec-top-badge">\u2605 Highest-Ranked Recommendation (#1 Top Pick)</span>'
                '<span class="rec-rank-pill">Best Match</span>'
            )
        else:
            rank_badges_html = (
                f'<span class="rec-alt-pill">Alternative Recommendation #{rank}</span>'
            )

        st.markdown(
            f"""
            <div class="{card_class}">
                <div class="rec-card-header">
                    <div>
                        <div>{rank_badges_html}</div>
                        <h3 class="rec-prog-title">{prog_name} <span style="font-weight: 500; color: #64748B;">({prog_code})</span></h3>
                        <p class="rec-prog-domain">
                            <strong>Domain:</strong> {domain} &nbsp;&bull;&nbsp;
                            <strong>Level:</strong> {level} &nbsp;&bull;&nbsp;
                            <strong>Duration:</strong> {duration}
                        </p>
                    </div>
                    <div class="rec-score-box">
                        <div class="rec-score-value">{overall:.1f} <span style="font-size: 0.85rem; font-weight: 600; color: #B91C1C;">/ 100</span></div>
                        <div class="rec-score-label">Match Score</div>
                    </div>
                </div>
                <div class="score-breakdown-grid">
                    <div class="score-subcard">
                        <div class="score-subcard-label">Current Programme Relevance (40%)</div>
                        <div class="score-subcard-val">{c_score:.1f} / 100</div>
                        <div class="score-bar-track"><div class="score-bar-fill" style="width: {min(100.0, max(0.0, c_score))}%;"></div></div>
                    </div>
                    <div class="score-subcard">
                        <div class="score-subcard-label">Professional Relevance (25%)</div>
                        <div class="score-subcard-val">{p_score:.1f} / 100</div>
                        <div class="score-bar-track"><div class="score-bar-fill" style="width: {min(100.0, max(0.0, p_score))}%;"></div></div>
                    </div>
                    <div class="score-subcard">
                        <div class="score-subcard-label">Experience &amp; Qualification Fit (20%)</div>
                        <div class="score-subcard-val">{e_score:.1f} / 100</div>
                        <div class="score-bar-track"><div class="score-bar-fill" style="width: {min(100.0, max(0.0, e_score))}%;"></div></div>
                    </div>
                    <div class="score-subcard">
                        <div class="score-subcard-label">Skill Progression (15%)</div>
                        <div class="score-subcard-val">{s_score:.1f} / 100</div>
                        <div class="score-bar-track"><div class="score-bar-fill" style="width: {min(100.0, max(0.0, s_score))}%;"></div></div>
                    </div>
                </div>
                <div class="ai-grid-3">
                    <div class="ai-subbox">
                        <div class="ai-subbox-title">Why It Matches the Student</div>
                        <p class="ai-subbox-text">{why_matches_text}</p>
                    </div>
                    <div class="ai-subbox">
                        <div class="ai-subbox-title">How It Builds on Their Existing Course</div>
                        <p class="ai-subbox-text">{builds_on_course_text}</p>
                    </div>
                    <div class="ai-subbox">
                        <div class="ai-subbox-title">Professional Relevance</div>
                        <p class="ai-subbox-text">{prof_relevance_text}</p>
                    </div>
                </div>
                <div class="ai-verify-box" style="margin-bottom: 0.9rem;">
                    <div class="ai-verify-title">Prerequisites &amp; Points to Verify</div>
                    <ul style="margin: 0; padding-left: 1.15rem; font-size: 0.85rem;">
                        {verify_html}
                    </ul>
                </div>
                <div class="ai-pitch-box" style="margin-bottom: 0.25rem;">
                    <div class="ai-pitch-title">Suggested Counsellor Pitch</div>
                    <p class="ai-pitch-text">&ldquo;{safe_pitch_text}&rdquo;</p>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Copy Pitch Button Row directly beneath each card
        copy_col1, copy_col2 = st.columns([1.4, 3.6], gap="small")
        with copy_col1:
            copy_clicked = st.button(
                f"\U0001f4cb Copy Pitch ({raw_code})",
                type="secondary",
                width="stretch",
                key=f"copy_pitch_btn_{rank}_{raw_code}",
            )
        if copy_clicked:
            st.session_state["copied_pitch_code"] = raw_code
            _copy_text_to_clipboard_js(raw_pitch_text)

        if st.session_state.get("copied_pitch_code") == raw_code:
            with copy_col2:
                st.success(f"Pitch for **{raw_code}** copied to clipboard (and ready to copy below):")
            st.code(raw_pitch_text, language=None)
        st.markdown("<div style='height: 0.35rem;'></div>", unsafe_allow_html=True)


