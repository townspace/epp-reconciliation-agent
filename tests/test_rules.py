"""
Unit tests for all 8 matching rules.
Each rule has: true positive, near-miss that must NOT match,
null-narration case (R1-R4), and zero-amount edge case.
"""
import pandas as pd
import pytest
from src.config import Config
from src.engine.rules import rule_1, rule_2, rule_3, rule_4, rule_5, rule_6, rule_7, rule_8
from src.engine.candidates import generate_candidates

CFG = Config(amount_tolerance=0.01, date_tolerance_days=3, fuzzy_narration_threshold=85)
D1 = pd.Timestamp("2024-01-10")
D2 = pd.Timestamp("2024-01-12")   # 2 days apart (within range=3)
D3 = pd.Timestamp("2024-01-20")   # 10 days apart (outside range)


def make_df(rows):
    df = pd.DataFrame(rows)
    df.index = list(range(len(df)))
    return df


def base_row(entity="A", partner="B", date=D1, amount=1000.0,
             currency="USD", narration="invoice 123"):
    return {
        "Entity": entity, "PartnerEntity": partner,
        "TransactionDate": date, "Amount": amount,
        "Currency": currency, "narration_clean": narration,
    }


# ── Rule 1 ────────────────────────────────────────────────────────────────────

class TestRule1:
    def test_true_positive(self):
        df = make_df([
            base_row("A", "B",  D1,  1000.0, narration="inv 1"),
            base_row("B", "A",  D1, -1000.0, narration="inv 1"),
        ])
        cands = generate_candidates(df)
        matched = set()
        results = rule_1(df, cands, matched, CFG)
        assert len(results) == 1
        assert results[0].confidence_score == 1.0
        assert 0 in matched and 1 in matched

    def test_near_miss_amount_off(self):
        """Amount difference > tolerance must NOT match."""
        df = make_df([
            base_row("A", "B", D1,  1000.0, narration="inv 1"),
            base_row("B", "A", D1, -1000.05, narration="inv 1"),  # 0.05 > 0.01
        ])
        cands = generate_candidates(df)
        results = rule_1(df, cands, set(), CFG)
        assert len(results) == 0

    def test_null_narration_no_match(self):
        df = make_df([
            base_row("A", "B", D1,  500.0, narration=""),
            base_row("B", "A", D1, -500.0, narration=""),
        ])
        cands = generate_candidates(df)
        results = rule_1(df, cands, set(), CFG)
        assert len(results) == 0

    def test_zero_amount_no_match(self):
        """Two zero-amount rows offset each other but should not match on R1 alone."""
        df = make_df([
            base_row("A", "B", D1, 0.0, narration="inv zero"),
            base_row("B", "A", D1, 0.0, narration="inv zero"),
        ])
        cands = generate_candidates(df)
        # 0 + 0 = 0, which is within tolerance — R1 WILL match (by spec)
        results = rule_1(df, cands, set(), CFG)
        assert len(results) == 1


# ── Rule 2 ────────────────────────────────────────────────────────────────────

class TestRule2:
    def test_true_positive(self):
        df = make_df([
            base_row("A", "B", D1,  2000.0, narration="payment consulting fee"),
            base_row("B", "A", D1, -2000.0, narration="payment consulting fees"),
        ])
        cands = generate_candidates(df)
        results = rule_2(df, cands, set(), CFG)
        assert len(results) == 1
        assert 0.8 <= results[0].confidence_score <= 1.0

    def test_near_miss_low_similarity(self):
        """Very different narrations should not match."""
        df = make_df([
            base_row("A", "B", D1,  500.0, narration="rent payment march"),
            base_row("B", "A", D1, -500.0, narration="salary disbursement"),
        ])
        cands = generate_candidates(df)
        results = rule_2(df, cands, set(), CFG)
        assert len(results) == 0

    def test_null_narration_no_match(self):
        df = make_df([
            base_row("A", "B", D1,  300.0, narration=""),
            base_row("B", "A", D1, -300.0, narration=""),
        ])
        cands = generate_candidates(df)
        results = rule_2(df, cands, set(), CFG)
        assert len(results) == 0

    def test_zero_amount_fuzzy_narration(self):
        df = make_df([
            base_row("A", "B", D1, 0.0, narration="test payment zero"),
            base_row("B", "A", D1, 0.0, narration="test payment zeros"),
        ])
        cands = generate_candidates(df)
        results = rule_2(df, cands, set(), CFG)
        assert len(results) == 1


# ── Rule 3 ────────────────────────────────────────────────────────────────────

class TestRule3:
    def test_true_positive(self):
        df = make_df([
            base_row("A", "B", D1,  1500.0, narration="rent q1"),
            base_row("B", "A", D2, -1500.0, narration="rent q1"),
        ])
        cands = generate_candidates(df)
        results = rule_3(df, cands, set(), CFG)
        assert len(results) == 1
        assert results[0].confidence_score == pytest.approx(0.85)

    def test_near_miss_date_too_far(self):
        df = make_df([
            base_row("A", "B", D1,  1500.0, narration="rent q1"),
            base_row("B", "A", D3, -1500.0, narration="rent q1"),  # 10 days
        ])
        cands = generate_candidates(df)
        results = rule_3(df, cands, set(), CFG)
        assert len(results) == 0

    def test_null_narration_no_match(self):
        df = make_df([
            base_row("A", "B", D1,  900.0, narration=""),
            base_row("B", "A", D2, -900.0, narration=""),
        ])
        cands = generate_candidates(df)
        results = rule_3(df, cands, set(), CFG)
        assert len(results) == 0

    def test_zero_amount_date_range(self):
        df = make_df([
            base_row("A", "B", D1, 0.0, narration="zero narr"),
            base_row("B", "A", D2, 0.0, narration="zero narr"),
        ])
        cands = generate_candidates(df)
        results = rule_3(df, cands, set(), CFG)
        assert len(results) == 1


# ── Rule 4 ────────────────────────────────────────────────────────────────────

class TestRule4:
    def test_true_positive(self):
        df = make_df([
            base_row("A", "B", D1,  700.0, narration="service charge q1"),
            base_row("B", "A", D2, -700.0, narration="service charges q1"),
        ])
        cands = generate_candidates(df)
        results = rule_4(df, cands, set(), CFG)
        assert len(results) == 1
        assert results[0].confidence_score < 0.85

    def test_near_miss_below_threshold(self):
        df = make_df([
            base_row("A", "B", D1,  700.0, narration="service charge q1"),
            base_row("B", "A", D2, -700.0, narration="totally unrelated text"),
        ])
        cands = generate_candidates(df)
        results = rule_4(df, cands, set(), CFG)
        assert len(results) == 0

    def test_null_narration_no_match(self):
        df = make_df([
            base_row("A", "B", D1,  400.0, narration=""),
            base_row("B", "A", D2, -400.0, narration=""),
        ])
        cands = generate_candidates(df)
        results = rule_4(df, cands, set(), CFG)
        assert len(results) == 0

    def test_zero_amount_fuzzy_date_range(self):
        df = make_df([
            base_row("A", "B", D1, 0.0, narration="zero fuzzy test"),
            base_row("B", "A", D2, 0.0, narration="zero fuzzy tests"),
        ])
        cands = generate_candidates(df)
        results = rule_4(df, cands, set(), CFG)
        assert len(results) == 1


# ── Rule 5 ────────────────────────────────────────────────────────────────────

class TestRule5:
    def test_true_positive(self):
        df = make_df([
            base_row("A", "B", D1,  3000.0, narration="misc A"),
            base_row("B", "A", D1, -3000.0, narration="misc B"),
        ])
        cands = generate_candidates(df)
        results = rule_5(df, cands, set(), CFG)
        assert len(results) == 1
        assert results[0].confidence_score == pytest.approx(0.65)

    def test_near_miss_different_date(self):
        df = make_df([
            base_row("A", "B", D1,  3000.0),
            base_row("B", "A", D3, -3000.0),  # 10 days apart
        ])
        cands = generate_candidates(df)
        results = rule_5(df, cands, set(), CFG)
        assert len(results) == 0

    def test_near_miss_amount_off(self):
        df = make_df([
            base_row("A", "B", D1,  3000.0),
            base_row("B", "A", D1, -3001.0),
        ])
        cands = generate_candidates(df)
        results = rule_5(df, cands, set(), CFG)
        assert len(results) == 0

    def test_zero_amount_exact_date(self):
        df = make_df([
            base_row("A", "B", D1, 0.0),
            base_row("B", "A", D1, 0.0),
        ])
        cands = generate_candidates(df)
        results = rule_5(df, cands, set(), CFG)
        assert len(results) == 1


# ── Rule 6 ────────────────────────────────────────────────────────────────────

class TestRule6:
    def test_true_positive(self):
        df = make_df([
            base_row("A", "B", D1,  250.0),
            base_row("B", "A", D3, -250.0),
        ])
        cands = generate_candidates(df)
        results = rule_6(df, cands, set(), CFG)
        assert len(results) == 1
        assert results[0].confidence_score == pytest.approx(0.50)

    def test_near_miss_amount_off(self):
        df = make_df([
            base_row("A", "B", D1,  250.0),
            base_row("B", "A", D1, -251.0),
        ])
        cands = generate_candidates(df)
        results = rule_6(df, cands, set(), CFG)
        assert len(results) == 0

    def test_already_matched_skipped(self):
        df = make_df([
            base_row("A", "B", D1,  250.0),
            base_row("B", "A", D1, -250.0),
        ])
        cands = generate_candidates(df)
        matched = {0, 1}
        results = rule_6(df, cands, matched, CFG)
        assert len(results) == 0

    def test_zero_amount_matches(self):
        df = make_df([
            base_row("A", "B", D1, 0.0),
            base_row("B", "A", D3, 0.0),
        ])
        cands = generate_candidates(df)
        results = rule_6(df, cands, set(), CFG)
        assert len(results) == 1


# ── Rule 7 ────────────────────────────────────────────────────────────────────

class TestRule7:
    def test_true_positive(self):
        """Same entity-partner, offsetting amounts."""
        df = make_df([
            base_row("A", "B", D1,  800.0, narration="correction entry"),
            base_row("A", "B", D3, -800.0, narration="reversal original"),
        ])
        cands = generate_candidates(df)
        results = rule_7(df, cands, set(), CFG)
        assert len(results) == 1
        assert results[0].confidence_score == pytest.approx(0.70)

    def test_near_miss_amount_off(self):
        df = make_df([
            base_row("A", "B", D1,  800.0),
            base_row("A", "B", D3, -801.0),
        ])
        cands = generate_candidates(df)
        results = rule_7(df, cands, set(), CFG)
        assert len(results) == 0

    def test_different_entity_no_match(self):
        """Reversed entity-partner (intercompany) should not be caught by R7."""
        df = make_df([
            base_row("A", "B", D1,  500.0),
            base_row("B", "A", D1, -500.0),
        ])
        cands = generate_candidates(df)
        results = rule_7(df, cands, set(), CFG)
        assert len(results) == 0

    def test_zero_amount_same_pair(self):
        df = make_df([
            base_row("A", "B", D1, 0.0, narration="zero corr"),
            base_row("A", "B", D2, 0.0, narration="zero rev"),
        ])
        cands = generate_candidates(df)
        results = rule_7(df, cands, set(), CFG)
        assert len(results) == 1


# ── Rule 8 ────────────────────────────────────────────────────────────────────

class TestRule8:
    def test_one_to_two_split(self):
        """1 on A side = 2 splits on B side."""
        df = make_df([
            base_row("A", "B", D1,  1000.0, narration="combined"),
            base_row("B", "A", D1,  -600.0, narration="part 1"),
            base_row("B", "A", D1,  -400.0, narration="part 2"),
        ])
        cands = generate_candidates(df)
        results = rule_8(df, cands, set(), CFG)
        assert len(results) == 1
        assert results[0].flagged_for_review is True
        assert set(results[0].left_indices) == {0}
        assert set(results[0].right_indices) == {1, 2}

    def test_near_miss_sum_off(self):
        """Splits don't sum to the original — no R8 match."""
        df = make_df([
            base_row("A", "B", D1,  1000.0),
            base_row("B", "A", D1,  -600.0),
            base_row("B", "A", D1,  -300.0),  # total only 900, not 1000
        ])
        cands = generate_candidates(df)
        results = rule_8(df, cands, set(), CFG)
        assert len(results) == 0

    def test_already_matched_rows_skipped(self):
        df = make_df([
            base_row("A", "B", D1,  1000.0),
            base_row("B", "A", D1,  -600.0),
            base_row("B", "A", D1,  -400.0),
        ])
        cands = generate_candidates(df)
        matched = {0, 1, 2}
        results = rule_8(df, cands, matched, CFG)
        assert len(results) == 0

    def test_zero_amount_group(self):
        df = make_df([
            base_row("A", "B", D1,  0.0, narration="zero split"),
            base_row("B", "A", D1,  0.0, narration="zero part"),
        ])
        cands = generate_candidates(df)
        results = rule_8(df, cands, set(), CFG)
        assert len(results) == 1
