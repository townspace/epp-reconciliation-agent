"""Tests for the orchestrator — no AI calls."""
import pandas as pd
import pytest
from src.config import Config
from src.engine.orchestrator import run_reconciliation
from src.models import ReconciliationReport

CFG = Config(amount_tolerance=0.01, date_tolerance_days=3, fuzzy_narration_threshold=85)
D1 = pd.Timestamp("2024-02-01")
D2 = pd.Timestamp("2024-02-03")


def _make_datasets(rows_a, rows_b):
    df_a = pd.DataFrame(rows_a)
    df_b = pd.DataFrame(rows_b)
    for df in (df_a, df_b):
        df["narration_clean"] = df["Narration"].str.lower().fillna("")
    return df_a, df_b


def base_row(entity, partner, date, amount, narration="payment"):
    return {"Entity": entity, "PartnerEntity": partner,
            "TransactionDate": date, "Amount": amount,
            "Currency": "USD", "Narration": narration}


class TestOrchestrator:
    def test_returns_reconciliation_report(self):
        df_a, df_b = _make_datasets(
            [base_row("A", "B", D1, 1000.0, "inv 1")],
            [base_row("B", "A", D1, -1000.0, "inv 1")],
        )
        report = run_reconciliation(df_a, df_b, CFG)
        assert isinstance(report, ReconciliationReport)

    def test_r1_match_captured(self):
        df_a, df_b = _make_datasets(
            [base_row("A", "B", D1, 1000.0, "inv 1")],
            [base_row("B", "A", D1, -1000.0, "inv 1")],
        )
        report = run_reconciliation(df_a, df_b, CFG)
        assert len(report.matched_pairs) == 1
        assert report.matched_pairs[0].rule.value == "R1_narr_exact_date_exact"

    def test_no_row_in_both_matched_and_unmatched(self):
        df_a, df_b = _make_datasets(
            [base_row("A", "B", D1, 500.0, "x"), base_row("A", "B", D2, 999.0)],
            [base_row("B", "A", D1, -500.0, "x")],
        )
        report = run_reconciliation(df_a, df_b, CFG)
        matched_set = set()
        for m in report.matched_pairs:
            matched_set.update(m.left_indices)
            matched_set.update(m.right_indices)
        for r8 in report.complex_groups:
            matched_set.update(r8.left_indices)
            matched_set.update(r8.right_indices)
        overlap = matched_set & set(report.unmatched_transactions)
        assert len(overlap) == 0

    def test_all_unmatched_when_no_pairs(self):
        df_a, df_b = _make_datasets(
            [base_row("A", "B", D1, 100.0)],
            [base_row("B", "A", D2, -999.99)],  # amount mismatch
        )
        report = run_reconciliation(df_a, df_b, CFG)
        assert len(report.matched_pairs) == 0
        assert len(report.unmatched_transactions) == 2

    def test_match_rate_100_percent(self):
        df_a, df_b = _make_datasets(
            [base_row("A", "B", D1, 300.0, "inv a")],
            [base_row("B", "A", D1, -300.0, "inv a")],
        )
        report = run_reconciliation(df_a, df_b, CFG)
        assert report.match_rate_pct == pytest.approx(100.0)

    def test_ai_fields_empty(self):
        df_a, df_b = _make_datasets(
            [base_row("A", "B", D1, 300.0)],
            [base_row("B", "A", D1, -300.0)],
        )
        report = run_reconciliation(df_a, df_b, CFG)
        assert report.anomaly_analysis == []
        assert report.complex_judgments == []
        assert report.executive_summary == ""
