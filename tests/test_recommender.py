"""Unit tests for the next-course recommendation engine in src/recommender.py."""
import unittest
import pandas as pd

from config import RECOMMENDATION_CONFIG
from src.data_loader import load_programs_data, load_students_data
from src.recommender import (
    CourseRecommender,
    extract_student_profile,
    filter_candidate_programs,
    normalize_weights,
    rank_candidate_programs,
    score_candidate_program,
    score_current_program_relevance,
    score_experience_and_qualification_fit,
    score_professional_relevance,
    score_skill_progression,
)


class TestRecommendationEngine(unittest.TestCase):
    """Comprehensive unit tests covering all recommendation engine steps and invariants."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.programs_df = load_programs_data(clean=True)
        cls.students_df = load_students_data(clean=True)
        cls.recommender = CourseRecommender(cls.programs_df, config=RECOMMENDATION_CONFIG)

    def test_configured_weights_sum_to_one(self) -> None:
        """Verify starting weights in config.py match 40%, 25%, 20%, 15% and sum to 1.0."""
        weights = normalize_weights()
        self.assertAlmostEqual(weights["current_program_relevance"], 0.40, places=4)
        self.assertAlmostEqual(weights["professional_relevance"], 0.25, places=4)
        self.assertAlmostEqual(weights["experience_qualification_fit"], 0.20, places=4)
        self.assertAlmostEqual(weights["skill_progression"], 0.15, places=4)
        self.assertAlmostEqual(sum(weights.values()), 1.0, places=6)

    def test_never_recommends_current_program(self) -> None:
        """Step 2 & 6: Ensure a student's current programme is never included in recommendations."""
        for _, student_row in self.students_df.iterrows():
            current_code = student_row["current_program_code"]
            recs = self.recommender.recommend_for_student(student_row, top_k=3)
            rec_codes = [r["program_code"] for r in recs]
            self.assertNotIn(
                current_code,
                rec_codes,
                f"Current programme {current_code} was erroneously recommended for {student_row['student_id']}",
            )

    def test_returns_top_three_with_required_schema_and_score_bounds(self) -> None:
        """Step 5: Verify top 3 results contain all required keys and scores out of 100."""
        required_keys = {
            "program_code",
            "program_name",
            "domain",
            "overall_match_score",
            "current_program_relevance_score",
            "professional_relevance_score",
            "experience_qualification_score",
            "skill_progression_score",
            "match_reasons",
            "potential_gaps",
            "prerequisites",
        }
        student_row = self.students_df.iloc[0]
        recs = self.recommender.recommend_for_student(student_row, top_k=3)
        self.assertEqual(len(recs), 3)

        # Verify descending sort order
        scores = [r["overall_match_score"] for r in recs]
        self.assertEqual(scores, sorted(scores, reverse=True))

        for rec in recs:
            self.assertTrue(required_keys.issubset(rec.keys()))
            for score_key in [
                "overall_match_score",
                "current_program_relevance_score",
                "professional_relevance_score",
                "experience_qualification_score",
                "skill_progression_score",
            ]:
                val = rec[score_key]
                self.assertGreaterEqual(val, 0.0)
                self.assertLessEqual(val, 100.0)
            self.assertIsInstance(rec["match_reasons"], list)
            self.assertIsInstance(rec["potential_gaps"], list)
            self.assertIsInstance(rec["prerequisites"], str)
            self.assertTrue(len(rec["prerequisites"]) > 0)

    def test_domain_filtering_respects_selected_domain(self) -> None:
        """Step 2: When a specific domain is selected, only programmes in that domain are returned."""
        student_row = self.students_df.iloc[0]
        target_domain = "AI & Generative AI"
        recs = self.recommender.recommend_for_student(
            student_row, top_k=3, preferred_domain=target_domain
        )
        self.assertGreaterEqual(len(recs), 1)
        for rec in recs:
            self.assertIn(target_domain.lower(), rec["domain"].lower())
            self.assertNotEqual(rec["program_code"], student_row["current_program_code"])

    def test_all_domains_considers_entire_catalogue(self) -> None:
        """Step 2: Selecting 'All Domains' considers the entire catalogue minus current programme."""
        student_row = self.students_df.iloc[0]
        candidates = filter_candidate_programs(
            self.programs_df,
            current_program_code=student_row["current_program_code"],
            preferred_domain="All Domains",
        )
        self.assertEqual(len(candidates), len(self.programs_df) - 1)

    def test_experience_suitability_and_counsellor_verification_flags(self) -> None:
        """Step 3: Experience below minimum does not crash or auto-reject, but lowers score and flags for counsellor verification."""
        junior_student = self.students_df[self.students_df["student_id"] == "STU047"].iloc[0]  # 0 yrs exp
        senior_student = self.students_df[self.students_df["student_id"] == "STU001"].iloc[0]  # 15 yrs exp

        junior_profile = extract_student_profile(junior_student, self.programs_df)
        senior_profile = extract_student_profile(senior_student, self.programs_df)

        # Pick CDAIO (requires 10+ years of experience and >50% marks)
        cdaio_row = self.programs_df[self.programs_df["program_code"] == "CDAIO"].iloc[0]

        j_score, _, j_gaps = score_experience_and_qualification_fit(junior_profile, cdaio_row)
        s_score, _, s_gaps = score_experience_and_qualification_fit(senior_profile, cdaio_row)

        self.assertGreater(s_score, j_score)
        self.assertTrue(
            any("Counsellor Verification (Experience)" in g for g in j_gaps),
            f"Expected experience verification flag in {j_gaps}",
        )
        self.assertTrue(
            any("Counsellor Verification (Prerequisites)" in g for g in s_gaps),
            f"Expected prerequisite verification flag for >50% score in {s_gaps}",
        )

    def test_missing_fields_and_unknown_current_program_graceful_handling(self) -> None:
        """Step 4 & 6: Missing student fields or unknown current_program_code do not crash or award 0 points, and surface a warning."""
        sparse_student = pd.Series(
            {
                "student_id": "STU_SPARSE",
                "current_program_code": "NON_EXISTENT_CODE",
                "enrollment_status": "completed",
                "completion_percentage": 100,
                "qualification": None,
                "degree_name": None,
                "completion_year": None,
                "experience_years": None,
                "designation": None,
                "working_status": None,
                "current_industry": None,
                "previous_industry": None,
            }
        )
        evaluation = rank_candidate_programs(
            student_record=sparse_student,
            programs_df=self.programs_df,
            top_k=3,
        )
        self.assertEqual(len(evaluation["recommendations"]), 3)
        self.assertLess(evaluation["data_coverage_ratio"], 0.5)
        self.assertGreaterEqual(len(evaluation["warnings"]), 2)

        for rec in evaluation["recommendations"]:
            self.assertGreater(rec["overall_match_score"], 0.0)
            self.assertGreater(rec["current_program_relevance_score"], 0.0)
            self.assertGreater(rec["professional_relevance_score"], 0.0)
            self.assertGreater(rec["experience_qualification_score"], 0.0)
            self.assertGreater(rec["skill_progression_score"], 0.0)

    def test_hr_professional_receives_hr_or_leadership_progression(self) -> None:
        """Verify domain and role alignment produces sensible top recommendation for HR Director."""
        hr_student = self.students_df[self.students_df["student_id"] == "STU014"].iloc[0]  # HR Director in DHRL
        recs = self.recommender.recommend_for_student(hr_student, top_k=3)
        top_codes = [r["program_code"] for r in recs]
        # Should include SHRL or IIML-CXO / CHRO / IIMR-CHRO
        hr_leadership_codes = {"SHRL", "IIML-CXO", "CHRO", "IIMR-CHRO"}
        self.assertTrue(
            bool(set(top_codes) & hr_leadership_codes),
            f"Expected at least one senior HR leadership programme in {top_codes}",
        )


if __name__ == "__main__":
    unittest.main()

