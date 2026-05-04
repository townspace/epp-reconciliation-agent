"""
8 deterministic matching rules using vectorised pandas operations.
Each function: rule_N(df_a, df_b, matched_a, matched_b, config) → List[MatchResult]
"""
import itertools
import logging
import uuid
from typing import List, Set

import pandas as pd
from rapidfuzz import fuzz

from src.config import Config
from src.models import MatchResult, MatchRule

logger = logging.getLogger(__name__)


# ── helpers ───────────────────────────────────────────────────────────────────

def _to_date(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce").dt.normalize()


def _unmatched(df: pd.DataFrame, matched: Set[int]) -> pd.DataFrame:
    return df[~df["_idx"].isin(matched)].copy()


def _make_result(idx_a: int, idx_b: int, rule: MatchRule, score: float) -> MatchResult:
    return MatchResult(
        recon_id=str(uuid.uuid4()),
        left_indices=[idx_a],
        right_indices=[idx_b],
        rule=rule,
        confidence_score=round(score, 4),
    )


def _greedy_dedupe(merged: pd.DataFrame, matched_a: Set[int], matched_b: Set[int],
                   col_a: str = "_idx_a", col_b: str = "_idx_b"):
    """
    Walk through merged matches greedily — each real index can only be used once.
    Returns list of (idx_a, idx_b) pairs.
    """
    pairs = []
    used_a, used_b = set(matched_a), set(matched_b)
    for _, row in merged.iterrows():
        a, b = int(row[col_a]), int(row[col_b])
        if a not in used_a and b not in used_b:
            pairs.append((a, b))
            used_a.add(a)
            used_b.add(b)
    return pairs


# ── Rule 1: exact narration + exact date + offset amount ──────────────────────

def rule_1(df_a, df_b, matched_a, matched_b, config):
    a = _unmatched(df_a, matched_a)
    b = _unmatched(df_b, matched_b)
    if a.empty or b.empty:
        return []

    a = a.copy(); b = b.copy()
    a["_date"] = _to_date(a["TransactionDate"])
    b["_date"] = _to_date(b["TransactionDate"])

    merged = pd.merge(
        a[["_idx", "narration_clean", "_date", "Amount"]],
        b[["_idx", "narration_clean", "_date", "Amount"]],
        on=["narration_clean", "_date"], suffixes=("_a", "_b"),
    )
    merged = merged[merged["narration_clean"] != ""]
    merged = merged[abs(merged["Amount_a"] + merged["Amount_b"]) <= config.amount_tolerance]

    pairs = _greedy_dedupe(merged.rename(columns={"_idx_a": "_idx_a", "_idx_b": "_idx_b"}), matched_a, matched_b)
    results = [_make_result(a, b, MatchRule.R1, 1.0) for a, b in pairs]
    for a, b in pairs:
        matched_a.add(a); matched_b.add(b)
    logger.info("R1: %d matches", len(results))
    return results


# ── Rule 2: fuzzy narration + exact date + offset amount ─────────────────────

def rule_2(df_a, df_b, matched_a, matched_b, config):
    a = _unmatched(df_a, matched_a)
    b = _unmatched(df_b, matched_b)
    if a.empty or b.empty:
        return []

    a = a.copy(); b = b.copy()
    a["_date"] = _to_date(a["TransactionDate"])
    b["_date"] = _to_date(b["TransactionDate"])

    # Merge on date + amount-offset, then fuzzy filter
    merged = pd.merge(
        a[["_idx", "narration_clean", "_date", "Amount"]],
        b[["_idx", "narration_clean", "_date", "Amount"]],
        on="_date", suffixes=("_a", "_b"),
    )
    merged = merged[abs(merged["Amount_a"] + merged["Amount_b"]) <= config.amount_tolerance]
    merged = merged[merged["narration_clean_a"] != ""]
    merged = merged[merged["narration_clean_b"] != ""]

    results, used_a, used_b = [], set(matched_a), set(matched_b)
    for _, row in merged.iterrows():
        ia, ib = int(row["_idx_a"]), int(row["_idx_b"])
        if ia in used_a or ib in used_b:
            continue
        ratio = fuzz.token_sort_ratio(row["narration_clean_a"], row["narration_clean_b"])
        if ratio >= config.fuzzy_narration_threshold:
            results.append(_make_result(ia, ib, MatchRule.R2, ratio / 100.0))
            used_a.add(ia); used_b.add(ib)
            matched_a.add(ia); matched_b.add(ib)

    logger.info("R2: %d matches", len(results))
    return results


# ── Rule 3: exact narration + date within range + offset amount ───────────────

def rule_3(df_a, df_b, matched_a, matched_b, config):
    a = _unmatched(df_a, matched_a)
    b = _unmatched(df_b, matched_b)
    if a.empty or b.empty:
        return []

    a = a.copy(); b = b.copy()
    a["_date"] = _to_date(a["TransactionDate"])
    b["_date"] = _to_date(b["TransactionDate"])

    merged = pd.merge(
        a[["_idx", "narration_clean", "_date", "Amount"]],
        b[["_idx", "narration_clean", "_date", "Amount"]],
        on="narration_clean", suffixes=("_a", "_b"),
    )
    merged = merged[merged["narration_clean"] != ""]
    merged = merged[abs(merged["Amount_a"] + merged["Amount_b"]) <= config.amount_tolerance]
    merged["_day_diff"] = abs((merged["_date_a"] - merged["_date_b"]).dt.days)
    merged = merged[
        (merged["_day_diff"] <= config.date_tolerance_days) &
        (merged["_day_diff"] > 0)   # exact date matches already handled by R1
    ]

    pairs = _greedy_dedupe(merged, matched_a, matched_b)
    results = [_make_result(a, b, MatchRule.R3, 0.85) for a, b in pairs]
    for a, b in pairs:
        matched_a.add(a); matched_b.add(b)
    logger.info("R3: %d matches", len(results))
    return results


# ── Rule 4: fuzzy narration + date within range + offset amount ───────────────

def rule_4(df_a, df_b, matched_a, matched_b, config):
    a = _unmatched(df_a, matched_a)
    b = _unmatched(df_b, matched_b)
    if a.empty or b.empty:
        return []

    a = a.copy(); b = b.copy()
    a["_date"] = _to_date(a["TransactionDate"])
    b["_date"] = _to_date(b["TransactionDate"])

    # Amount-bucket merge instead of cross-join: b offset ≈ -a offset
    tol = max(config.amount_tolerance, 0.01)
    a["_amt_key"] = (a["Amount"] / tol).round()
    b["_amt_key"] = (-b["Amount"] / tol).round()

    merged = pd.merge(
        a[["_idx", "narration_clean", "_date", "Amount", "_amt_key"]].rename(
            columns={"_idx": "_idx_a", "narration_clean": "narr_a", "_date": "_date_a", "Amount": "Amount_a"}),
        b[["_idx", "narration_clean", "_date", "Amount", "_amt_key"]].rename(
            columns={"_idx": "_idx_b", "narration_clean": "narr_b", "_date": "_date_b", "Amount": "Amount_b"}),
        on="_amt_key",
    )
    merged = merged[abs(merged["Amount_a"] + merged["Amount_b"]) <= config.amount_tolerance]
    merged["_day_diff"] = abs((merged["_date_a"] - merged["_date_b"]).dt.days)
    merged = merged[merged["_day_diff"] <= config.date_tolerance_days]
    merged = merged[merged["narr_a"] != ""]
    merged = merged[merged["narr_b"] != ""]

    results, used_a, used_b = [], set(matched_a), set(matched_b)
    for _, row in merged.iterrows():
        ia, ib = int(row["_idx_a"]), int(row["_idx_b"])
        if ia in used_a or ib in used_b:
            continue
        ratio = fuzz.token_sort_ratio(row["narr_a"], row["narr_b"])
        if ratio >= config.fuzzy_narration_threshold:
            results.append(_make_result(ia, ib, MatchRule.R4, 0.75 * ratio / 100.0))
            used_a.add(ia); used_b.add(ib)
            matched_a.add(ia); matched_b.add(ib)

    logger.info("R4: %d matches", len(results))
    return results


# ── Rule 5: exact date + offset amount (no narration) ────────────────────────

def rule_5(df_a, df_b, matched_a, matched_b, config):
    a = _unmatched(df_a, matched_a)
    b = _unmatched(df_b, matched_b)
    if a.empty or b.empty:
        return []

    a = a.copy(); b = b.copy()
    a["_date"] = _to_date(a["TransactionDate"])
    b["_date"] = _to_date(b["TransactionDate"])

    merged = pd.merge(
        a[["_idx", "_date", "Amount"]],
        b[["_idx", "_date", "Amount"]],
        on="_date", suffixes=("_a", "_b"),
    )
    merged = merged[abs(merged["Amount_a"] + merged["Amount_b"]) <= config.amount_tolerance]

    pairs = _greedy_dedupe(merged, matched_a, matched_b)
    results = [_make_result(a, b, MatchRule.R5, 0.65) for a, b in pairs]
    for a, b in pairs:
        matched_a.add(a); matched_b.add(b)
    logger.info("R5: %d matches", len(results))
    return results


# ── Rule 6: amount offset only ────────────────────────────────────────────────

def rule_6(df_a, df_b, matched_a, matched_b, config):
    a = _unmatched(df_a, matched_a)
    b = _unmatched(df_b, matched_b)
    if a.empty or b.empty:
        return []

    # Amount-bucket merge instead of cross-join
    tol = max(config.amount_tolerance, 0.01)
    a = a.copy(); b = b.copy()
    a["_amt_key"] = (a["Amount"] / tol).round()
    b["_amt_key"] = (-b["Amount"] / tol).round()

    merged = pd.merge(
        a[["_idx", "Amount", "_amt_key"]].rename(columns={"_idx": "_idx_a", "Amount": "Amount_a"}),
        b[["_idx", "Amount", "_amt_key"]].rename(columns={"_idx": "_idx_b", "Amount": "Amount_b"}),
        on="_amt_key",
    )
    merged = merged[abs(merged["Amount_a"] + merged["Amount_b"]) <= config.amount_tolerance]

    pairs = _greedy_dedupe(merged, matched_a, matched_b)
    results = [_make_result(a, b, MatchRule.R6, 0.50) for a, b in pairs]
    for a, b in pairs:
        matched_a.add(a); matched_b.add(b)
    logger.info("R6: %d matches", len(results))
    return results


# ── Rule 7: reversal detection (same entity-pair, offset amount) ──────────────

def rule_7(df_a, df_b, matched_a, matched_b, config):
    """Same entity AND partner (not reversed) — finds same-side corrections."""
    a = _unmatched(df_a, matched_a)
    b = _unmatched(df_b, matched_b)
    # R7 targets same-direction pairs — look within df_a itself
    um_a = df_a[~df_a["_idx"].isin(matched_a)].copy()
    if len(um_a) < 2:
        return []

    merged = pd.merge(
        um_a[["_idx", "Entity", "PartnerEntity", "Amount"]].add_suffix("_x").rename(columns={"_idx_x": "_idx_x"}),
        um_a[["_idx", "Entity", "PartnerEntity", "Amount"]].add_suffix("_y").rename(columns={"_idx_y": "_idx_y"}),
        left_on=["Entity_x", "PartnerEntity_x"],
        right_on=["Entity_y", "PartnerEntity_y"],
    )
    merged = merged[merged["_idx_x"] < merged["_idx_y"]]
    merged = merged[abs(merged["Amount_x"] + merged["Amount_y"]) <= config.amount_tolerance]

    results, used_a = [], set(matched_a)
    for _, row in merged.iterrows():
        ia, ib = int(row["_idx_x"]), int(row["_idx_y"])
        if ia in used_a or ib in used_a:
            continue
        results.append(_make_result(ia, ib, MatchRule.R7, 0.70))
        used_a.add(ia); used_a.add(ib)
        matched_a.add(ia); matched_a.add(ib)

    logger.info("R7: %d matches", len(results))
    return results


# ── Rule 8: many-to-many grouping ─────────────────────────────────────────────

def rule_8(df_a, df_b, matched_a, matched_b, config) -> List[MatchResult]:
    tol = config.amount_tolerance
    um_a = df_a[~df_a["_idx"].isin(matched_a)].copy()
    um_b = df_b[~df_b["_idx"].isin(matched_b)].copy()
    if um_a.empty or um_b.empty:
        return []

    # Pre-build a fast lookup: _idx → Amount for bank side
    b_amt: dict = dict(zip(um_b["_idx"], um_b["Amount"]))
    b_idx_list = [i for i in b_amt if not pd.isna(b_amt[i])]

    results = []
    used_a, used_b = set(), set()

    for _, row_a in um_a.iterrows():
        ia = int(row_a["_idx"])
        if ia in used_a:
            continue
        amt_a = row_a["Amount"]
        if pd.isna(amt_a):
            continue

        # Pre-filter: only bank rows whose amount has the right sign/magnitude
        # abs(amt_a) should roughly equal sum of candidates
        avail = [i for i in b_idx_list if i not in used_b]
        if not avail:
            break

        # For size-1: quick vectorised check first (most common case)
        found = False
        for ib in avail:
            if abs(amt_a + b_amt[ib]) <= tol:
                results.append(MatchResult(
                    recon_id=str(uuid.uuid4()),
                    left_indices=[ia],
                    right_indices=[ib],
                    rule=MatchRule.R8,
                    confidence_score=0.60,
                    flagged_for_review=True,
                ))
                used_a.add(ia); used_b.add(ib)
                matched_a.add(ia); matched_b.add(ib)
                found = True
                break

        if found:
            continue

        # Size 2-3: only try candidates within plausible amount range
        # amt_a + sum(combo) ≈ 0  →  sum(combo) ≈ -amt_a
        target = -amt_a
        # Filter to candidates whose individual amount is smaller than |target|
        narrowed = [i for i in avail
                    if abs(b_amt[i]) < abs(target) + tol and
                       abs(b_amt[i]) > tol][:50]  # cap at 50

        for size in range(2, 4):  # sizes 2 and 3 only
            if found or len(narrowed) < size:
                break
            for combo in itertools.combinations(narrowed, size):
                combo_sum = sum(b_amt[i] for i in combo)
                if abs(amt_a + combo_sum) <= tol:
                    results.append(MatchResult(
                        recon_id=str(uuid.uuid4()),
                        left_indices=[ia],
                        right_indices=list(combo),
                        rule=MatchRule.R8,
                        confidence_score=0.60,
                        flagged_for_review=True,
                    ))
                    used_a.add(ia)
                    used_b.update(combo)
                    matched_a.add(ia)
                    matched_b.update(combo)
                    found = True
                    break

    logger.info("R8: %d groups", len(results))
    return results
