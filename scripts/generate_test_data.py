"""
Generate tests/fixtures/sample_input.xlsx with synthetic reconciliation data.

Layout:
  Sheet "dataset_1" — Side A transactions
  Sheet "dataset_2" — Side B transactions (counterparts)

Breakdown (50 total transactions across 5 entity pairs):
  10  R1 matches  — exact narration, exact date, offsetting amounts
   5  R2 matches  — fuzzy narration (~90% similarity), exact date, offsetting
   5  R3 matches  — exact narration, ±2 days, offsetting
   3  R7 reversals — same entity-pair, offsetting, no narration match
   2  R8 groups   — 1-to-2 splits (1 tx on A side = 2 on B side)
  10  genuine unmatched — no counterpart on the other side
  15  noise rows  — partial matches (amount off, narration off, etc.)
"""

import os
import random
import pandas as pd
from datetime import date, timedelta
from pathlib import Path

random.seed(42)

ENTITY_PAIRS = [
    ("Alpha Corp", "Beta Ltd"),
    ("Gamma Inc", "Delta Co"),
    ("Epsilon LLC", "Zeta GmbH"),
    ("Eta SA", "Theta BV"),
    ("Iota AG", "Kappa KK"),
]
CURRENCIES = ["USD", "USD", "EUR", "USD", "JPY"]
BASE_DATE = date(2024, 3, 1)


def rand_date(offset_days=0):
    return BASE_DATE + timedelta(days=random.randint(0, 60) + offset_days)


def _row(entity, partner, txdate, amount, currency, narration):
    return {
        "Entity": entity,
        "PartnerEntity": partner,
        "TransactionDate": txdate,
        "Amount": round(amount, 2),
        "Currency": currency,
        "Narration": narration,
    }


rows_a, rows_b = [], []

# ── R1 matches: exact narration, exact date, offsetting ──────────────────────
for i in range(13):  # 13 pairs → 26 rows matched at R1
    ep = ENTITY_PAIRS[i % 5]
    cur = CURRENCIES[i % 5]
    d = rand_date()
    amt = round(random.uniform(1000, 50000), 2)
    narr = f"INV-{1000 + i} service payment Q1"
    rows_a.append(_row(ep[0], ep[1], d,  amt, cur, narr))
    rows_b.append(_row(ep[1], ep[0], d, -amt, cur, narr))

# ── R2 matches: fuzzy narration (~1-2 char diff), exact date ─────────────────
for i in range(5):
    ep = ENTITY_PAIRS[i % 5]
    cur = CURRENCIES[i % 5]
    d = rand_date()
    amt = round(random.uniform(500, 20000), 2)
    narr_a = f"PYMT ref {2000 + i} consulting fee"
    narr_b = f"PYMT ref {2000 + i} consulting fees"   # extra 's' — high similarity
    rows_a.append(_row(ep[0], ep[1], d,  amt, cur, narr_a))
    rows_b.append(_row(ep[1], ep[0], d, -amt, cur, narr_b))

# ── R3 matches: exact narration, ±2 days ─────────────────────────────────────
for i in range(5):
    ep = ENTITY_PAIRS[i % 5]
    cur = CURRENCIES[i % 5]
    d = rand_date()
    amt = round(random.uniform(200, 15000), 2)
    narr = f"RENT-{3000 + i} monthly rental"
    rows_a.append(_row(ep[0], ep[1], d,            amt, cur, narr))
    rows_b.append(_row(ep[1], ep[0], d + timedelta(days=2), -amt, cur, narr))

# ── R7 reversals: same entity-pair, offsetting, generic narration ─────────────
for i in range(3):
    ep = ENTITY_PAIRS[i % 5]
    cur = CURRENCIES[i % 5]
    d = rand_date()
    amt = round(random.uniform(100, 5000), 2)
    rows_a.append(_row(ep[0], ep[1], d,  amt, cur, f"REVERSAL correction {4000 + i}"))
    rows_b.append(_row(ep[1], ep[0], d, -amt, cur, f"REVERSAL correction {4000 + i}"))

# ── R8 groups: 1 on A side splits into 2 on B side ───────────────────────────
for i in range(2):
    ep = ENTITY_PAIRS[i % 5]
    cur = CURRENCIES[i % 5]
    d = rand_date()
    amt_total = round(random.uniform(5000, 30000), 2)
    split1 = round(amt_total * 0.6, 2)
    split2 = round(amt_total - split1, 2)
    rows_a.append(_row(ep[0], ep[1], d,       amt_total, cur, f"SPLIT-{5000 + i} combined payment"))
    rows_b.append(_row(ep[1], ep[0], d,  -split1,       cur, f"SPLIT-{5000 + i} part 1"))
    rows_b.append(_row(ep[1], ep[0], d,  -split2,       cur, f"SPLIT-{5000 + i} part 2"))

# ── 10 genuinely unmatched (A side only) ─────────────────────────────────────
for i in range(10):
    ep = ENTITY_PAIRS[i % 5]
    cur = CURRENCIES[i % 5]
    d = rand_date()
    amt = round(random.uniform(50, 10000), 2)
    rows_a.append(_row(ep[0], ep[1], d, amt, cur, f"UNMATCHED-{6000 + i} no counterpart"))

# ── 15 noise rows spread across both sides ───────────────────────────────────
for i in range(5):   # A side noise
    ep = ENTITY_PAIRS[i % 5]
    cur = CURRENCIES[i % 5]
    d = rand_date()
    amt = round(random.uniform(100, 9999), 2)
    rows_a.append(_row(ep[0], ep[1], d, amt, cur, f"NOISE-A-{7000 + i}"))

for i in range(5):   # B side noise
    ep = ENTITY_PAIRS[i % 5]
    cur = CURRENCIES[i % 5]
    d = rand_date()
    amt = round(random.uniform(100, 9999), 2)
    rows_b.append(_row(ep[1], ep[0], d, -amt, cur, f"NOISE-B-{8000 + i}"))

df_a = pd.DataFrame(rows_a)
df_b = pd.DataFrame(rows_b)

out_path = Path(__file__).parent.parent / "tests" / "fixtures" / "sample_input.xlsx"
out_path.parent.mkdir(parents=True, exist_ok=True)

with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
    df_a.to_excel(writer, sheet_name="dataset_1", index=False)
    df_b.to_excel(writer, sheet_name="dataset_2", index=False)

print(f"Generated {len(df_a)} rows (dataset_1) + {len(df_b)} rows (dataset_2)")
print(f"Saved to: {out_path}")
