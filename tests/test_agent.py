"""
Agent tests — all OpenAI API calls are mocked.
"""
import json
import pytest
import pandas as pd
from unittest.mock import MagicMock, patch
from openai import RateLimitError

from src.config import Config
from src.models import AnomalyResult, ComplexMatchJudgment, MatchResult, MatchRule
from src.agent.agent import analyze_unmatched, score_complex_match, generate_summary

CFG = Config(
    openai_api_key="test-key",
    skip_ai=False,
    amount_tolerance=0.01,
    date_tolerance_days=3,
    fuzzy_narration_threshold=85,
)
CFG_SKIP = Config(openai_api_key="", skip_ai=True)


def _mock_client(text: str):
    """Return a mock OpenAI client whose completions return `text`."""
    choice = MagicMock()
    choice.message.content = text
    response = MagicMock()
    response.choices = [choice]
    client = MagicMock()
    client.chat.completions.create.return_value = response
    return client


def _make_group(left_idx=0, right_idx=1):
    return MatchResult(
        recon_id="group-abc-123",
        left_indices=[left_idx],
        right_indices=[right_idx],
        rule=MatchRule.R8,
        confidence_score=0.60,
        flagged_for_review=True,
    )


def _make_df():
    return pd.DataFrame([
        {"Entity": "A", "PartnerEntity": "B", "Amount": 1000.0,
         "TransactionDate": pd.Timestamp("2024-01-01"), "narration_clean": "payment"},
        {"Entity": "B", "PartnerEntity": "A", "Amount": -1000.0,
         "TransactionDate": pd.Timestamp("2024-01-01"), "narration_clean": "payment"},
    ])


# ── analyze_unmatched ─────────────────────────────────────────────────────────

class TestAnalyzeUnmatched:
    def test_returns_anomaly_results(self):
        payload = json.dumps([
            {"transaction_index": 5, "classification": "MISSING_COUNTERPART",
             "explanation": "No matching entry found.", "suggested_action": "Investigate with partner entity."},
        ])
        with patch("src.agent.agent.OpenAI", return_value=_mock_client(payload)):
            results = analyze_unmatched(
                [{"index": 5, "entity": "A", "amount": 100.0}], CFG
            )
        assert len(results) == 1
        assert isinstance(results[0], AnomalyResult)
        assert results[0].classification == "MISSING_COUNTERPART"
        assert results[0].transaction_index == 5

    def test_skip_ai_returns_empty(self):
        results = analyze_unmatched([{"index": 1}], CFG_SKIP)
        assert results == []

    def test_retry_on_rate_limit_error(self):
        """Mock RateLimitError on attempt 1, success on attempt 2."""
        payload = json.dumps([
            {"transaction_index": 3, "classification": "UNKNOWN",
             "explanation": "Could not determine.", "suggested_action": "Manual review."},
        ])
        call_count = 0

        def side_effect(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RateLimitError(
                    message="rate limit",
                    response=MagicMock(status_code=429, headers={}),
                    body={},
                )
            choice = MagicMock()
            choice.message.content = payload
            resp = MagicMock()
            resp.choices = [choice]
            return resp

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = side_effect

        with patch("src.agent.agent.OpenAI", return_value=mock_client):
            with patch("src.agent.agent.time.sleep"):
                results = analyze_unmatched([{"index": 3}], CFG)

        assert call_count == 2
        assert len(results) == 1


# ── score_complex_match ───────────────────────────────────────────────────────

class TestScoreComplexMatch:
    @pytest.mark.parametrize("verdict", ["LIKELY_VALID", "NEEDS_REVIEW", "LIKELY_ERROR"])
    def test_handles_all_verdicts(self, verdict):
        payload = json.dumps({
            "group_id": "group-abc-123",
            "verdict": verdict,
            "reasoning": "Test reasoning.",
            "confidence": 0.8,
        })
        with patch("src.agent.agent.OpenAI", return_value=_mock_client(payload)):
            result = score_complex_match(_make_group(), _make_df(), CFG)
        assert isinstance(result, ComplexMatchJudgment)
        assert result.verdict == verdict
        assert result.group_id == "group-abc-123"

    def test_skip_ai_returns_needs_review(self):
        result = score_complex_match(_make_group(), _make_df(), CFG_SKIP)
        assert result.verdict == "NEEDS_REVIEW"
        assert result.confidence == 0.0

    def test_bad_json_returns_fallback(self):
        with patch("src.agent.agent.OpenAI", return_value=_mock_client("not json")):
            result = score_complex_match(_make_group(), _make_df(), CFG)
        assert result.verdict == "NEEDS_REVIEW"


# ── generate_summary ──────────────────────────────────────────────────────────

class TestGenerateSummary:
    def _make_report(self):
        from src.models import ReconciliationReport
        from datetime import datetime, timezone
        return ReconciliationReport(
            run_id="run-xyz",
            run_timestamp=datetime.now(timezone.utc),
            total_transactions=10,
            matched_pairs=[],
            unmatched_transactions=[0, 1],
            complex_groups=[],
            anomaly_analysis=[],
            complex_judgments=[],
            executive_summary="",
            match_rate_pct=80.0,
        )

    def test_returns_string(self):
        with patch("src.agent.agent.OpenAI", return_value=_mock_client("Great summary.")):
            result = generate_summary(self._make_report(), CFG)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_skip_ai_returns_fallback(self):
        result = generate_summary(self._make_report(), CFG_SKIP)
        assert "Reconciliation Run" in result
        assert "Match rate" in result
