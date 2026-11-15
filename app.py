"""Streamlit Frontend for the Student Next-Course Recommendation System."""
from typing import Dict, Any, Optional
import pandas as pd
import streamlit as st

from config import BASE_DIR, MIN_COMPLETION_PERCENTAGE, RECOMMENDATION_CONFIG
from src.data_loader import (
    load_programs_data,
    load_students_data,
    get_recommendation_queue,
)
from src.data_validator import validate_all_datasets
from src.recommender import CourseRecommender
from src.ai_explainer import AIExplainer
from src.ui_components import (
    load_custom_css,
    get_unique_domains_from_programs,
    render_app_header,
    render_summary_metric_cards,
    render_student_filters,
    render_student_table,
    render_student_profile_cards,
    render_recommendation_results,
    format_display_value,
)

st.set_page_config(
    page_title="Student Next-Course Recommendation System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data(show_spinner=False)
def load_and_validate_application_data() -> Dict[str, Any]:
    """Load, clean, and validate students and programmes datasets."""
    programs_df = load_programs_data(clean=True)
    students_df = load_students_data(clean=True)
    min_threshold = float(
        RECOMMENDATION_CONFIG.get("min_completion_threshold", MIN_COMPLETION_PERCENTAGE)
    )
    eligible_df = get_recommendation_queue(
        students_df, min_completion_percentage=min_threshold
    )
    validation_report = validate_all_datasets(
        students_df,
        programs_df,
        min_completion_percentage=min_threshold,
        emit_warning=False,
    )
    return {
        "programs_df": programs_df,
        "students_df": students_df,
        "eligible_df": eligible_df,
        "validation_report": validation_report,
        "min_threshold": min_threshold,
    }


def navigate_to_student_recommendation(student_id: str) -> None:
    """Switch active view to Screen 2 (Student Recommendation) for the specified student."""
    st.session_state["selected_student_id"] = student_id
    st.session_state["active_screen"] = "Student Recommendation"
    st.session_state["recommendation_result"] = None
    st.session_state["ai_explanation_result"] = None
    st.session_state["copied_pitch_code"] = None
    st.rerun()


def render_dashboard_screen(data_bundle: Dict[str, Any]) -> None:
    """Render Screen 1: Dashboard with 4 summary metric cards, search/filters, and eligible student table."""
    programs_df: pd.DataFrame = data_bundle["programs_df"]
    eligible_df: pd.DataFrame = data_bundle["eligible_df"]
    min_threshold: float = data_bundle["min_threshold"]

    total_eligible = len(eligible_df)
    nearing_completion_count = int(
        (eligible_df["enrollment_status"] == "in_progress").sum()
    )
    completed_count = int(
        (eligible_df["enrollment_status"] == "completed").sum()
    )
    available_programmes_count = len(programs_df)

    # 1. Four Summary Metric Cards
    render_summary_metric_cards(
        total_students=total_eligible,
        nearing_completion_count=nearing_completion_count,
        completed_count=completed_count,
        available_programmes_count=available_programmes_count,
        min_completion_threshold=min_threshold,
    )

    st.markdown("<div style='height: 0.75rem;'></div>", unsafe_allow_html=True)

    # 2. Search & Filters (only on eligible students)
    filtered_df = render_student_filters(eligible_df)

    # 3. Quick Student Action Bar + Eligible Student Table
    with st.container():
        q_col1, q_col2 = st.columns([3.5, 1.5], gap="small")
        student_options = filtered_df["student_id"].dropna().astype(str).tolist()
        with q_col1:
            quick_selected_id = st.selectbox(
                "Quick Select Student for Counselling",
                options=student_options if student_options else ["No matching students"],
                index=0,
                format_func=lambda sid: (
                    sid
                    if sid == "No matching students"
                    else _format_student_selector_option(filtered_df, sid)
                ),
                key="dashboard_quick_student_select",
            )
        with q_col2:
            st.markdown("<div style='height: 1.72rem;'></div>", unsafe_allow_html=True)
            if st.button(
                "Open Student Profile \u2192",
                type="primary",
                width="stretch",
                disabled=not bool(student_options),
                key="dashboard_open_student_btn",
            ):
                navigate_to_student_recommendation(quick_selected_id)

    render_student_table(
        filtered_df=filtered_df,
        total_eligible_count=total_eligible,
        on_open_student=navigate_to_student_recommendation,
    )


def _format_student_selector_option(df: pd.DataFrame, student_id: str) -> str:
    """Format a student option as 'STU001 — Designation (Current Programme | Status)'."""
    match = df[df["student_id"] == student_id]
    if match.empty:
        return student_id
    row = match.iloc[0]
    prog = format_display_value(row.get("current_program_code"), fallback="N/A")
    desig = format_display_value(row.get("designation"), fallback="No Designation")
    status = "Completed" if row.get("enrollment_status") == "completed" else f"In Progress ({int(row.get('completion_percentage', 0))}%)"
    return f"{student_id} — {desig} [{prog} • {status}]"


def _find_program_record(programs_df: pd.DataFrame, program_code: Optional[str]) -> Optional[pd.Series]:
    """Look up a student's current programme in programs_df."""
    if not program_code or programs_df.empty or "program_code" not in programs_df.columns:
        return None
    matches = programs_df[programs_df["program_code"] == program_code]
    if matches.empty:
        return None
    return matches.iloc[0]


def render_student_recommendation_screen(data_bundle: Dict[str, Any]) -> None:
    """Render Screen 2: Student Recommendation with student profile, optional domain selector, and action button."""
    programs_df: pd.DataFrame = data_bundle["programs_df"]
    eligible_df: pd.DataFrame = data_bundle["eligible_df"]

    if eligible_df.empty:
        st.warning("No eligible students available in the recommendation queue.")
        return

    student_ids = eligible_df["student_id"].dropna().astype(str).tolist()
    current_selected_id = st.session_state.get("selected_student_id")
    if current_selected_id not in student_ids:
        current_selected_id = student_ids[0]
        st.session_state["selected_student_id"] = current_selected_id

    # Top Student Selection & Back Navigation Bar
    top_col1, top_col2 = st.columns([1.2, 3.8], gap="small")
    with top_col1:
        st.markdown("<div style='height: 1.72rem;'></div>", unsafe_allow_html=True)
        if st.button(
            "\u2190 Back to Dashboard",
            type="secondary",
            width="stretch",
            key="back_to_dashboard_btn",
        ):
            st.session_state["active_screen"] = "Dashboard"
            st.rerun()

    with top_col2:
        selected_idx = student_ids.index(current_selected_id)
        chosen_student_id = st.selectbox(
            "Select Eligible Student",
            options=student_ids,
            index=selected_idx,
            format_func=lambda sid: _format_student_selector_option(eligible_df, sid),
            key="rec_screen_student_selector",
        )
        if chosen_student_id != current_selected_id:
            st.session_state["selected_student_id"] = chosen_student_id
            st.session_state["recommendation_result"] = None
            st.session_state["ai_explanation_result"] = None
            st.session_state["copied_pitch_code"] = None
            st.rerun()

    student_row = eligible_df[eligible_df["student_id"] == chosen_student_id].iloc[0]
    current_prog_code = student_row.get("current_program_code")
    program_record = _find_program_record(programs_df, current_prog_code)

    missing_program_warning = None
    if program_record is None:
        missing_program_warning = (
            f"Warning: Current programme code '{current_prog_code or 'MISSING'}' for student "
            f"'{chosen_student_id}' is not present in the programme catalogue (programs.csv). "
            "No assumptions are being made about the current programme's curriculum."
        )

    # Render Student Academic & Professional Profile Cards
    render_student_profile_cards(
        student_row=student_row,
        program_record=program_record,
        missing_program_warning=missing_program_warning,
    )

    # Optional Domain Selector & Generate Recommendations Action Card
    st.markdown(
        """
        <div class="section-header" style="margin-top: 0.5rem; margin-bottom: 0.75rem;">
            <div class="section-title">
                <span class="section-title-accent"></span>
                <span>Next-Course Recommendation Controls</span>
            </div>
            <span class="section-badge">Optional Domain Filter</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    domain_options = get_unique_domains_from_programs(programs_df)

    ctrl_col1, ctrl_col2 = st.columns([3.2, 1.8], gap="medium")
    with ctrl_col1:
        selected_domain = st.selectbox(
            "Explore programmes in a specific domain",
            options=domain_options,
            index=0,
            help="Optional: Filter candidate recommendations to a specific domain from programs.csv, or leave as 'All Domains'.",
            key="rec_domain_selector",
        )
    with ctrl_col2:
        st.markdown("<div style='height: 1.72rem;'></div>", unsafe_allow_html=True)
        generate_clicked = st.button(
            "Generate Recommendations",
            type="primary",
            width="stretch",
            key="generate_recommendations_btn",
        )

    def _run_recommendation_and_ai_pipeline() -> None:
        """Steps 8, 9 & 10: Run Python recommendation engine to rank top 3 programmes,
        then invoke GenAI layer to generate reasons and suggested pitching angles.
        """
        st.session_state["copied_pitch_code"] = None
        recommender = CourseRecommender(programs_df=programs_df, config=RECOMMENDATION_CONFIG)
        eval_payload = recommender.evaluate_student_recommendations(
            student_record=student_row,
            top_k=int(RECOMMENDATION_CONFIG.get("top_k", 3)),
            preferred_domain=selected_domain,
        )
        st.session_state["recommendation_result"] = eval_payload

        recs = eval_payload.get("recommendations", [])
        if recs:
            explainer = AIExplainer()
            if explainer.is_configured():
                with st.spinner("Generating recommendations and AI counselling pitches..."):
                    st.session_state["ai_explanation_result"] = (
                        explainer.generate_explanations_for_recommendations(
                            student_record=student_row,
                            recommendations=recs,
                            programs_df=programs_df,
                        )
                    )
            else:
                st.session_state["ai_explanation_result"] = (
                    explainer.generate_explanations_for_recommendations(
                        student_record=student_row,
                        recommendations=recs,
                        programs_df=programs_df,
                    )
                )
        else:
            st.session_state["ai_explanation_result"] = None

    if generate_clicked:
        _run_recommendation_and_ai_pipeline()

    recommendation_result = st.session_state.get("recommendation_result")
    if recommendation_result and recommendation_result.get("student_id") == chosen_student_id:
        def _handle_generate_again() -> None:
            _run_recommendation_and_ai_pipeline()
            st.rerun()

        def _handle_retry_ai_only() -> None:
            recs = recommendation_result.get("recommendations", [])
            if recs:
                explainer = AIExplainer()
                with st.spinner("Retrying Groq counselling explanations and pitches..."):
                    st.session_state["ai_explanation_result"] = (
                        explainer.generate_explanations_for_recommendations(
                            student_record=student_row,
                            recommendations=recs,
                            programs_df=programs_df,
                        )
                    )
                st.rerun()

        # Step 11: Display final unified recommendation cards while student profile remains visible above
        render_recommendation_results(
            evaluation_payload=recommendation_result,
            student_row=student_row,
            programs_df=programs_df,
            ai_result=st.session_state.get("ai_explanation_result"),
            on_generate_again=_handle_generate_again,
            on_retry_ai=_handle_retry_ai_only,
        )


def main() -> None:
    """Main entrypoint for the Streamlit UI."""
    load_custom_css(BASE_DIR / "assets" / "styles.css")

    if "active_screen" not in st.session_state:
        st.session_state["active_screen"] = "Dashboard"
    if "selected_student_id" not in st.session_state:
        st.session_state["selected_student_id"] = None
    if "recommendation_result" not in st.session_state:
        st.session_state["recommendation_result"] = None
    if "ai_explanation_result" not in st.session_state:
        st.session_state["ai_explanation_result"] = None
    if "copied_pitch_code" not in st.session_state:
        st.session_state["copied_pitch_code"] = None

    try:
        data_bundle = load_and_validate_application_data()
    except Exception as exc:
        render_app_header("Student Next-Course Recommendation System")
        st.error(f"Failed to load application data: {exc}")
        return

    validation_report = data_bundle["validation_report"]

    # Sidebar Navigation
    with st.sidebar:
        st.markdown(
            """
            <div style="padding: 0.25rem 0 1rem 0; border-bottom: 1px solid #E2E8F0; margin-bottom: 1rem;">
                <div style="display: flex; align-items: center; gap: 0.65rem;">
                    <div class="saas-brand-badge" style="width: 34px; height: 34px; font-size: 1rem;">A</div>
                    <div>
                        <div style="font-weight: 700; font-size: 0.95rem; color: #0F172A;">Accredian</div>
                        <div style="font-size: 0.75rem; color: #64748B;">Counselling Console</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        screens = ["Dashboard", "Student Recommendation"]
        current_screen_idx = (
            screens.index(st.session_state["active_screen"])
            if st.session_state["active_screen"] in screens
            else 0
        )

        chosen_screen = st.radio(
            "WORKSPACE NAVIGATION",
            options=screens,
            index=current_screen_idx,
            key="sidebar_nav_radio",
        )
        if chosen_screen != st.session_state["active_screen"]:
            st.session_state["active_screen"] = chosen_screen
            st.rerun()

        st.markdown("<hr style='margin: 1.25rem 0; border-color: #E2E8F0;'>", unsafe_allow_html=True)
        st.caption("QUEUE ELIGIBILITY RULES")
        st.markdown(
            f"""
            <div style="font-size: 0.8rem; color: #475569; background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px; padding: 0.75rem;">
                <div style="margin-bottom: 0.35rem;">&bull; <strong>Completed:</strong> 100% eligible</div>
                <div style="margin-bottom: 0.35rem;">&bull; <strong>In Progress:</strong> &ge; {data_bundle['min_threshold']:.0f}% completion</div>
                <div>&bull; <strong>Withdrawn:</strong> Excluded</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if validation_report["unmatched_student_programs"]:
            st.markdown("<div style='height: 0.75rem;'></div>", unsafe_allow_html=True)
            st.warning(
                f"{len(validation_report['unmatched_student_programs'])} student(s) have a programme code not found in programs.csv."
            )

    # Top Header
    render_app_header("Student Next-Course Recommendation System")

    # Display global dataset warnings if any student's current_program_code is missing from catalogue
    if validation_report["unmatched_student_programs"]:
        with st.expander("⚠️ Catalogue Cross-Reference Warnings", expanded=True):
            for item in validation_report["unmatched_student_programs"]:
                st.warning(item["warning"])

    # Screen Router
    if st.session_state["active_screen"] == "Dashboard":
        render_dashboard_screen(data_bundle)
    else:
        render_student_recommendation_screen(data_bundle)


if __name__ == "__main__":
    main()
