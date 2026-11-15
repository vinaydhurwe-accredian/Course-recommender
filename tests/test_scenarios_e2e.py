"""Comprehensive end-to-end test suite covering all 12 scenarios and validation checks in Prompt 7."""
import json
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock
import pandas as pd

from config import RECOMMENDATION_CONFIG
from src.data_loader import (
    get_recommendation_queue,
    is_student_eligible_for_recommendation,
    load_programs_data,
    load_students_data,
)
from src.data_validator import (
    find_duplicate_program_codes,
    find_students_with_missing_program_in_catalog,
    validate_all_datasets,
)
from src.recommender import (
    CourseRecommender,
    filter_candidate_programs,
    rank_candidate_programs,
    score_candidate_program,
    extract_student_profile,
)
from src.ai_explainer import (
    AIExplainer,
    build_deterministic_explanation_for_program,
)


class TestEndToEndScenarios(unittest.TestCase):
    """Test all 12 operational scenarios and system-wide invariants."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.programs_df = load_programs_data(clean=True)
        cls.students_df = load_students_data(clean=True)
        cls.recommender = CourseRecommender(cls.programs_df, config=RECOMMENDATION_CONFIG)

    def test_datasets_load_and_validate_correctly(self) -> None:
        """Validation: Student records and programme records load and validate cleanly."""
        self.assertGreater(len(self.programs_df), 30)
        self.assertGreater(len(self.students_df), 40)
        report = validate_all_datasets(self.students_df, self.programs_df, emit_warning=False)
        self.assertTrue(report["is_valid"])
        self.assertEqual(len(report["duplicate_program_codes"]), 0)
        self.assertEqual(len(report["duplicate_student_ids"]), 0)

    def test_duplicate_program_codes_are_detected(self) -> None:
        """Validation: Duplicate programme codes are detected by data_validator."""
        dup_df = pd.concat([self.programs_df.iloc[:2], self.programs_df.iloc[:1]], ignore_index=True)
        dups = find_duplicate_program_codes(dup_df)
        self.assertIn(self.programs_df.iloc[0]["program_code"], dups)

    def test_scenario_01_student_completed_current_programme(self) -> None:
        """Scenario 1: Student has completed their current programme -> eligible and receives top 3 ranked recommendations."""
        completed_student = self.students_df[
            self.students_df["enrollment_status"] == "completed"
        ].iloc[0]
        self.assertTrue(is_student_eligible_for_recommendation(completed_student))

        eval_payload = self.recommender.evaluate_student_recommendations(completed_student, top_k=3)
        recs = eval_payload["recommendations"]
        self.assertEqual(len(recs), 3)
        rec_codes = [r["program_code"] for r in recs]
        self.assertNotIn(completed_student["current_program_code"], rec_codes)

    def test_scenario_02_student_in_progress_95_percent(self) -> None:
        """Scenario 2: Student is in progress with 95% completion -> eligible (>=80%) and receives top 3 recommendations."""
        student_95 = self.students_df[
            (self.students_df["enrollment_status"] == "in_progress")
            & (self.students_df["completion_percentage"] == 95)
        ].iloc[0]
        self.assertTrue(is_student_eligible_for_recommendation(student_95))

        queue_df = get_recommendation_queue(self.students_df)
        self.assertIn(student_95["student_id"], queue_df["student_id"].tolist())

        recs = self.recommender.recommend_for_student(student_95, top_k=3)
        self.assertEqual(len(recs), 3)
        self.assertNotIn(student_95["current_program_code"], [r["program_code"] for r in recs])

    def test_scenario_03_student_in_progress_50_percent_or_withdrawn(self) -> None:
        """Scenario 3: Student is in progress with 50% completion (or withdrawn) -> excluded from eligible queue."""
        student_50 = pd.Series(
            {
                "student_id": "STU_50_PCT",
                "current_program_code": "IIMV-EPLAI",
                "enrollment_status": "in_progress",
                "completion_percentage": 50.0,
                "qualification": "MBA",
                "degree_name": "MBA",
                "completion_year": 2018,
                "experience_years": 6,
                "designation": "Manager",
                "working_status": "yes",
                "current_industry": "IT",
                "previous_industry": "BFSI",
            }
        )
        student_withdrawn = student_50.copy()
        student_withdrawn["student_id"] = "STU_WITHDRAWN"
        student_withdrawn["enrollment_status"] = "withdrawn"
        student_withdrawn["completion_percentage"] = 95.0

        self.assertFalse(is_student_eligible_for_recommendation(student_50))
        self.assertFalse(is_student_eligible_for_recommendation(student_withdrawn))

        combined_df = pd.DataFrame([student_50, student_withdrawn])
        queue = get_recommendation_queue(combined_df)
        self.assertTrue(queue.empty)

    def test_scenario_04_student_missing_professional_information(self) -> None:
        """Scenario 4: Student has missing professional info (STU047 or sparse) -> no crash, scores > 0, flags verification."""
        stu047 = self.students_df[self.students_df["student_id"] == "STU047"].iloc[0]
        eval_payload = self.recommender.evaluate_student_recommendations(stu047, top_k=3)
        recs = eval_payload["recommendations"]
        self.assertEqual(len(recs), 3)
        for rec in recs:
            self.assertGreater(rec["overall_match_score"], 0.0)
            self.assertGreater(rec["professional_relevance_score"], 0.0)
            # Verify fallback explanation also works cleanly
            exp = build_deterministic_explanation_for_program(stu047, rec, self.programs_df)
            self.assertTrue(exp["why_this_programme"])
            self.assertTrue(exp["suggested_pitch"])

    def test_scenario_05_student_current_program_missing_from_catalog(self) -> None:
        """Scenario 5: Student's current_program_code is missing from programs.csv -> emits warning, no curriculum assumed."""
        unmatched_student = pd.Series(
            {
                "student_id": "STU_UNMATCHED_PROG",
                "current_program_code": "UNKNOWN-CODE-2026",
                "enrollment_status": "completed",
                "completion_percentage": 100,
                "qualification": "MBA",
                "degree_name": "MBA - Finance",
                "completion_year": 2012,
                "experience_years": 12,
                "designation": "VP - Strategy",
                "working_status": "yes",
                "current_industry": "BFSI",
                "previous_industry": "Consulting",
            }
        )
        unmatched_report = find_students_with_missing_program_in_catalog(
            pd.DataFrame([unmatched_student]), self.programs_df, emit_warning=False
        )
        self.assertEqual(len(unmatched_report), 1)

        eval_payload = self.recommender.evaluate_student_recommendations(unmatched_student, top_k=3)
        self.assertTrue(
            any("UNKNOWN-CODE-2026" in w for w in eval_payload["warnings"]),
            f"Expected warning for missing programme code in {eval_payload['warnings']}",
        )
        self.assertEqual(len(eval_payload["recommendations"]), 3)
        exp = build_deterministic_explanation_for_program(
            unmatched_student, eval_payload["recommendations"][0], self.programs_df
        )
        self.assertIn("not listed in programs.csv", exp["learning_progression"])

    def test_scenario_06_user_selects_all_domains(self) -> None:
        """Scenario 6: User selects All Domains -> considers all programmes except student's current programme."""
        student = self.students_df.iloc[0]
        filtered = filter_candidate_programs(
            self.programs_df,
            current_program_code=student["current_program_code"],
            preferred_domain="All Domains",
        )
        self.assertEqual(len(filtered), len(self.programs_df) - 1)
        eval_payload = self.recommender.evaluate_student_recommendations(
            student, top_k=3, preferred_domain="All Domains"
        )
        self.assertEqual(len(eval_payload["recommendations"]), 3)

    def test_scenario_07_user_selects_specific_domain(self) -> None:
        """Scenario 7: User selects a specific domain -> only programmes matching that domain are recommended."""
        student = self.students_df.iloc[0]
        domain = "Project Management & Leadership"
        eval_payload = self.recommender.evaluate_student_recommendations(
            student, top_k=3, preferred_domain=domain
        )
        recs = eval_payload["recommendations"]
        self.assertGreaterEqual(len(recs), 1)
        for rec in recs:
            self.assertIn(domain.lower(), rec["domain"].lower())

    def test_scenario_08_no_suitable_programmes_in_selected_domain(self) -> None:
        """Scenario 8: No suitable programmes available in selected domain (e.g. student is already in the only programme in that domain)."""
        # STU033 is enrolled in CFO, which is the only programme in the "Finance" domain
        stu_cfo = self.students_df[self.students_df["student_id"] == "STU033"].iloc[0]
        eval_payload = self.recommender.evaluate_student_recommendations(
            stu_cfo, top_k=3, preferred_domain="Finance"
        )
        self.assertEqual(len(eval_payload["recommendations"]), 0)
        self.assertTrue(
            any("No candidate programmes matched" in w for w in eval_payload["warnings"])
        )

    def test_scenario_09_student_has_no_experience_recorded(self) -> None:
        """Scenario 9: Student has no experience recorded (None/NaN or 0 years) -> handled gracefully with counsellor verification flag."""
        no_exp_student = self.students_df.iloc[0].copy()
        no_exp_student["experience_years"] = None
        eval_none = self.recommender.evaluate_student_recommendations(no_exp_student, top_k=3)
        self.assertEqual(len(eval_none["recommendations"]), 3)
        for rec in eval_none["recommendations"]:
            self.assertTrue(
                any("experience" in g.lower() for g in rec["potential_gaps"]),
                f"Expected experience verification flag in {rec['potential_gaps']}",
            )

        # Also check 0 years experience (STU047)
        stu047 = self.students_df[self.students_df["student_id"] == "STU047"].iloc[0]
        eval_zero = self.recommender.evaluate_student_recommendations(stu047, top_k=3)
        self.assertEqual(len(eval_zero["recommendations"]), 3)

    def test_scenario_10_groq_api_key_is_missing(self) -> None:
        """Scenario 10: Groq API key is missing -> app does not crash, returns structured missing_key status."""
        student = self.students_df.iloc[0]
        recs = self.recommender.recommend_for_student(student, top_k=3)
        explainer = AIExplainer(api_key="", model="openai/gpt-oss-120b")
        res = explainer.generate_explanations_for_recommendations(
            student_record=student, recommendations=recs, programs_df=self.programs_df
        )
        self.assertFalse(res["success"])
        self.assertEqual(res["error_type"], "missing_key")
        self.assertEqual(len(recs), 3)

    def test_scenario_11_groq_api_returns_error(self) -> None:
        """Scenario 11: Groq API returns an error -> app catches exception, preserves recommendations, allows retry."""
        student = self.students_df.iloc[0]
        recs = self.recommender.recommend_for_student(student, top_k=3)
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception("503 Service Unavailable")

        explainer = AIExplainer(api_key="gsk-valid-looking-test-key", model="openai/gpt-oss-120b", client=mock_client)
        res = explainer.generate_explanations_for_recommendations(
            student_record=student, recommendations=recs, programs_df=self.programs_df
        )
        self.assertFalse(res["success"])
        self.assertEqual(res["error_type"], "api_error")
        self.assertIn("503 Service Unavailable", res["error"])

    def test_scenario_12_programme_has_incomplete_information(self) -> None:
        """Scenario 12: A programme has incomplete information (e.g., IIM-TRICHY-CTO missing skills_covered & min_experience_years)."""
        incomplete_prog = self.programs_df[
            self.programs_df["program_code"] == "IIM-TRICHY-CTO"
        ].iloc[0]
        self.assertTrue(pd.isna(incomplete_prog["skills_covered"]))
        self.assertTrue(pd.isna(incomplete_prog["minimum_experience_years"]))

        student = self.students_df.iloc[0]
        profile = extract_student_profile(student, self.programs_df)
        scored = score_candidate_program(profile, incomplete_prog)
        self.assertGreater(scored["overall_match_score"], 0.0)
        self.assertGreater(scored["skill_progression_score"], 0.0)
        self.assertGreater(scored["experience_qualification_score"], 0.0)
        self.assertTrue(scored["prerequisites"])


if __name__ == "__main__":
    unittest.main()

