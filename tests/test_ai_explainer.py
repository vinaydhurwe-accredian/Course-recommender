"""Unit tests for the GenAI explanation and pitching layer in src/ai_explainer.py."""
import json
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock
import pandas as pd

from src.data_loader import load_programs_data, load_students_data
from src.recommender import CourseRecommender
from src.ai_explainer import (
    AIExplainer,
    ALLOWED_STUDENT_FIELDS,
    build_deterministic_explanation_for_program,
    build_privacy_safe_api_payload,
    sanitize_student_profile_for_ai,
    sanitize_shortlisted_programs_for_ai,
)


class TestAIExplainer(unittest.TestCase):
    """Verify privacy guardrails, prompt grounding, structured outputs, and error handling."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.programs_df = load_programs_data(clean=True)
        cls.students_df = load_students_data(clean=True)
        cls.recommender = CourseRecommender(cls.programs_df)

    def test_privacy_sanitization_strips_all_pii_and_identifiers(self) -> None:
        """Verify that student_id, names, emails, phones, and sensitive demographics are never included."""
        student_with_pii = pd.Series(
            {
                "student_id": "STU001",
                "student_name": "Rahul Sharma",
                "full_name": "Rahul Sharma",
                "email": "rahul.sharma@example.com",
                "mobile_number": "+91-9876543210",
                "phone": "9876543210",
                "gender": "Male",
                "date_of_birth": "1985-04-12",
                "address": "Mumbai, India",
                "aadhaar_number": "1234-5678-9012",
                "current_program_code": "IIMV-EPLAI",
                "enrollment_status": "completed",
                "completion_percentage": 100,
                "qualification": "MBA",
                "degree_name": "MBA - Finance",
                "completion_year": 2008,
                "experience_years": 15,
                "designation": "Senior Manager - Digital Transformation",
                "working_status": "yes",
                "current_industry": "BFSI",
                "previous_industry": "IT",
            }
        )

        recs = self.recommender.recommend_for_student(student_with_pii, top_k=3)
        payload = build_privacy_safe_api_payload(
            student_record=student_with_pii,
            recommendations=recs,
            programs_df=self.programs_df,
        )
        serialized = json.dumps(payload)

        # Ensure forbidden values are absent
        self.assertNotIn("STU001", serialized)
        self.assertNotIn("Rahul Sharma", serialized)
        self.assertNotIn("rahul.sharma@example.com", serialized)
        self.assertNotIn("9876543210", serialized)
        self.assertNotIn("1985-04-12", serialized)
        self.assertNotIn("1234-5678-9012", serialized)

        # Ensure only allowed student keys + catalog/missing metadata exist
        profile_part = payload["student_academic_and_professional_profile"]
        for key in ALLOWED_STUDENT_FIELDS:
            self.assertIn(key, profile_part)
        self.assertNotIn("student_id", profile_part)
        self.assertNotIn("student_name", profile_part)
        self.assertNotIn("email", profile_part)
        self.assertNotIn("mobile_number", profile_part)

    def test_missing_or_placeholder_api_key_handled_gracefully(self) -> None:
        """Application must not crash when GROQ_API_KEY is missing or set to placeholder."""
        student_row = self.students_df.iloc[0]
        recs = self.recommender.recommend_for_student(student_row, top_k=3)

        explainer_empty = AIExplainer(api_key="", model="openai/gpt-oss-120b")
        self.assertEqual(explainer_empty.base_url, "https://api.groq.com/openai/v1")
        self.assertFalse(explainer_empty.is_configured())
        res_empty = explainer_empty.generate_explanations_for_recommendations(
            student_record=student_row,
            recommendations=recs,
            programs_df=self.programs_df,
        )
        self.assertFalse(res_empty["success"])
        self.assertEqual(res_empty["error_type"], "missing_key")
        self.assertIn("GROQ_API_KEY", res_empty["error"])

        explainer_placeholder = AIExplainer(
            api_key="your_groq_api_key_here", model="openai/gpt-oss-120b"
        )
        self.assertFalse(explainer_placeholder.is_configured())
        res_placeholder = explainer_placeholder.generate_explanations_for_recommendations(
            student_record=student_row,
            recommendations=recs,
            programs_df=self.programs_df,
        )
        self.assertFalse(res_placeholder["success"])
        self.assertEqual(res_placeholder["error_type"], "missing_key")

    def test_successful_structured_explanation_generation_and_no_invented_programs(self) -> None:
        """Verify all 5 required sections are returned per recommended programme and invented codes are ignored."""
        student_row = self.students_df.iloc[0]
        recs = self.recommender.recommend_for_student(student_row, top_k=3)
        rec_codes = [r["program_code"] for r in recs]

        mock_response_payload = {
            "explanations": [
                {
                    "program_code": rec_codes[0],
                    "why_this_programme": "Matches the student's MBA in Finance and 15 years of experience in BFSI.",
                    "learning_progression": "Builds on IIMV-EPLAI by expanding into enterprise digital and AI governance.",
                    "professional_relevance": "Directly aligns with the Senior Manager - Digital Transformation role in BFSI.",
                    "suggested_pitch": "Having completed IIMV-EPLAI, this 11-month programme is a practical next step for your digital transformation remit.",
                    "points_to_verify": [
                        "Verify minimum 50% marks in graduation.",
                        "Confirm readiness for 11-month executive schedule.",
                    ],
                },
                {
                    "program_code": rec_codes[1],
                    "why_this_programme": "Relevant for senior strategy and AI leadership progression.",
                    "learning_progression": "Extends foundational AI strategy into corporate governance and boardroom communication.",
                    "professional_relevance": "Supports senior managerial progression in BFSI and IT.",
                    "suggested_pitch": "Given your background in BFSI and IT, this programme strengthens strategic execution.",
                    "points_to_verify": ["Verify 5+ years of work experience documentation."],
                },
                {
                    "program_code": rec_codes[2],
                    "why_this_programme": "Connects executive management with technology transformation.",
                    "learning_progression": "Adds cross-functional leadership and financial strategy modules.",
                    "professional_relevance": "Relevant to senior transformation managers in financial services.",
                    "suggested_pitch": "This programme complements your technical AI training with enterprise leadership.",
                    "points_to_verify": ["Confirm graduate degree transcripts."],
                },
                {
                    "program_code": "INVENTED-FAKE-CODE-999",
                    "why_this_programme": "Should be filtered out.",
                    "learning_progression": "Should be filtered out.",
                    "professional_relevance": "Should be filtered out.",
                    "suggested_pitch": "Should be filtered out.",
                    "points_to_verify": ["Should be filtered out."],
                },
            ]
        }

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=json.dumps(mock_response_payload))
                )
            ]
        )

        explainer = AIExplainer(
            api_key="gsk-test-valid-key-12345",
            model="openai/gpt-oss-120b",
            client=mock_client,
        )
        result = explainer.generate_explanations_for_recommendations(
            student_record=student_row,
            recommendations=recs,
            programs_df=self.programs_df,
        )

        self.assertTrue(result["success"])
        self.assertIsNone(result["error"])
        explanations = result["explanations"]

        # Invented programme code must be rejected
        self.assertNotIn("INVENTED-FAKE-CODE-999", explanations)
        self.assertEqual(set(explanations.keys()), set(rec_codes))

        for code in rec_codes:
            item = explanations[code]
            self.assertTrue(item["why_this_programme"])
            self.assertTrue(item["learning_progression"])
            self.assertTrue(item["professional_relevance"])
            self.assertTrue(item["suggested_pitch"])
            self.assertIsInstance(item["points_to_verify"], list)
            self.assertGreaterEqual(len(item["points_to_verify"]), 1)

    def test_api_failure_does_not_crash_and_preserves_recommendations(self) -> None:
        """If the Groq API raises an exception, AIExplainer returns a clean error dict without crashing."""
        student_row = self.students_df.iloc[0]
        recs = self.recommender.recommend_for_student(student_row, top_k=3)
        original_Snapshot = json.dumps(recs, sort_keys=True)

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = RuntimeError("Connection timeout to Groq API")

        explainer = AIExplainer(
            api_key="gsk-test-valid-key-12345",
            model="openai/gpt-oss-120b",
            client=mock_client,
        )
        result = explainer.generate_explanations_for_recommendations(
            student_record=student_row,
            recommendations=recs,
            programs_df=self.programs_df,
        )

        self.assertFalse(result["success"])
        self.assertEqual(result["error_type"], "api_error")
        self.assertIn("Connection timeout", result["error"])
        # Ensure original recommendations list and rankings were never mutated
        self.assertEqual(json.dumps(recs, sort_keys=True), original_Snapshot)

    def test_deterministic_fallback_explanation_contains_all_required_card_fields(self) -> None:
        """Verify that build_deterministic_explanation_for_program provides all 5 required card fields."""
        student_row = self.students_df.iloc[0]
        recs = self.recommender.recommend_for_student(student_row, top_k=3)
        for rec in recs:
            fallback = build_deterministic_explanation_for_program(
                student_record=student_row,
                rec=rec,
                programs_df=self.programs_df,
            )
            self.assertEqual(fallback["program_code"], rec["program_code"])
            self.assertTrue(fallback["why_this_programme"])
            self.assertTrue(fallback["learning_progression"])
            self.assertTrue(fallback["professional_relevance"])
            self.assertTrue(fallback["suggested_pitch"])
            self.assertIsInstance(fallback["points_to_verify"], list)
            self.assertGreaterEqual(len(fallback["points_to_verify"]), 1)


if __name__ == "__main__":
    unittest.main()

