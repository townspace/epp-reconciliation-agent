"""
Pre-build candidate index with date-bucketed pre-filtering for performance.
"""
import logging
from typing import Dict, List, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

# Max candidates per row — prevents O(n²) explosion on large datasets
MAX_CANDIDATES = 500


def generate_candidates(df: pd.DataFrame, date_tolerance_days: int = 5) -> Dict[int, List[int]]:
    """
    Returns dict: row index → list of candidate row indices.

    Pre-filters by:
      1. Entity-partner relationship (reversed pair for intercompany / bank recon)
      2. Date proximity (±date_tolerance_days * 2) to avoid full cross-product
    """
    candidates: Dict[int, List[int]] = {i: [] for i in df.index}

    # Build lookup: (Entity, PartnerEntity) → list of row indices
    pair_index: Dict[Tuple[str, str], List[int]] = {}
    for idx, row in df.iterrows():
        key = (str(row["Entity"]), str(row["PartnerEntity"]))
        pair_index.setdefault(key, []).append(idx)

    # Pre-extract dates for fast comparison
    dates = {}
    for idx, row in df.iterrows():
        d = row.get("TransactionDate")
        dates[idx] = pd.Timestamp(d) if pd.notna(d) else None

    # Date window: be generous (tolerance * 3) to catch R3/R4
    date_window = pd.Timedelta(days=max(date_tolerance_days * 3, 10))

    for idx, row in df.iterrows():
        entity  = str(row["Entity"])
        partner = str(row["PartnerEntity"])
        d_a     = dates[idx]

        cands = []

        # Reversed pair (main intercompany / bank recon matches)
        reversed_key = (partner, entity)
        for other_idx in pair_index.get(reversed_key, []):
            if other_idx == idx:
                continue
            d_b = dates[other_idx]
            # Include if dates within window OR either date is NaT
            if d_a is None or d_b is None or abs((d_a - d_b).days) <= date_window.days:
                cands.append(other_idx)

        # Same-side pair (reversals/corrections)
        same_key = (entity, partner)
        for other_idx in pair_index.get(same_key, []):
            if other_idx == idx:
                continue
            d_b = dates[other_idx]
            if d_a is None or d_b is None or abs((d_a - d_b).days) <= date_window.days:
                cands.append(other_idx)

        # Deduplicate
        seen, deduped = set(), []
        for c in cands:
            if c not in seen:
                seen.add(c)
                deduped.append(c)

        # Hard cap to prevent combinatorial explosion
        if len(deduped) > MAX_CANDIDATES:
            logger.debug("Row %d: capping candidates from %d to %d", idx, len(deduped), MAX_CANDIDATES)
            deduped = deduped[:MAX_CANDIDATES]

        candidates[idx] = deduped

    total_cands = sum(len(v) for v in candidates.values())
    logger.info("Candidate index: %d rows, %d total candidates (avg %.1f/row)",
                len(df), total_cands, total_cands / max(len(df), 1))
    return candidates
