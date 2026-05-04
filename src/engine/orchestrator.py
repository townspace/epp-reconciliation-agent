"""
Sequential rule runner — NEVER calls the AI agent.
Returns a partially-filled ReconciliationReport (AI fields empty).
"""
import logging
import uuid
from datetime import datetime, timezone

import pandas as pd

from src.config import Config
from src.engine.rules import rule_1, rule_2, rule_3, rule_4, rule_5, rule_6, rule_7, rule_8
from src.models import ReconciliationReport

logger = logging.getLogger(__name__)

_RULES = [rule_1, rule_2, rule_3, rule_4, rule_5, rule_6, rule_7]


def run_reconciliation(df_1: pd.DataFrame, df_2: pd.DataFrame, config: Config) -> ReconciliationReport:
    df_1 = df_1.copy()
    df_2 = df_2.copy()
    df_1["side"] = "A"
    df_2["side"] = "B"

    # Add internal integer index column for tracking
    df_1["_idx"] = range(len(df_1))
    df_2["_idx"] = range(len(df_1), len(df_1) + len(df_2))

    total = len(df_1) + len(df_2)
    logger.info("Starting reconciliation: %d total rows (%d ledger, %d bank)", total, len(df_1), len(df_2))

    matched_a: set = set()   # _idx values from df_1
    matched_b: set = set()   # _idx values from df_2
    matched_pairs = []

    # R1 → R7
    for rule_fn in _RULES:
        results = rule_fn(df_1, df_2, matched_a, matched_b, config)
        matched_pairs.extend(results)
        logger.info("%-10s matched %d pairs (total matched indices: %d)",
                    rule_fn.__name__, len(results), len(matched_a) + len(matched_b))

    # R8 on remaining
    r8_results = rule_8(df_1, df_2, matched_a, matched_b, config)
    complex_groups = r8_results

    # Collect unmatched
    all_idx = set(df_1["_idx"]) | set(df_2["_idx"])
    r8_idx  = set()
    for r in r8_results:
        r8_idx.update(r.left_indices)
        r8_idx.update(r.right_indices)

    unmatched_idx = sorted(all_idx - matched_a - matched_b - r8_idx)

    # Build df_all for downstream use (exporter, AI agent)
    df_all = pd.concat([df_1, df_2], ignore_index=True)

    run_id = str(uuid.uuid4())
    total_matched = len(matched_a) + len(matched_b) + len(r8_idx)
    match_rate = (total_matched / total * 100) if total > 0 else 0.0

    logger.info(
        "Done: run_id=%s matched=%d unmatched=%d r8_groups=%d match_rate=%.1f%%",
        run_id, total_matched, len(unmatched_idx), len(complex_groups), match_rate,
    )

    return ReconciliationReport(
        run_id=run_id,
        run_timestamp=datetime.now(timezone.utc),
        total_transactions=total,
        matched_pairs=matched_pairs,
        unmatched_transactions=unmatched_idx,
        complex_groups=complex_groups,
        anomaly_analysis=[],
        complex_judgments=[],
        executive_summary="",
        match_rate_pct=round(match_rate, 2),
    )
