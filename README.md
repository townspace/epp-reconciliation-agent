# EPP Reconciliation AI Agent

A fully automated Python AI agent that ingests intercompany Excel datasets, runs an 8-rule sequential matching engine, applies Claude AI judgment at three specific decision points, and outputs a fully reconciled Excel report with audit trail.

## Installation

```bash
pip install -e '.[test]'
cp .env.example .env
# Edit .env and set your ANTHROPIC_API_KEY
```

## Usage

```bash
# Full run with AI
python run.py --input IC_InputData.xlsx --output ./output

# Skip AI (deterministic only)
python run.py --input IC_InputData.xlsx --output ./output --skip-ai

# With overrides and verbose logging
python run.py --input IC_InputData.xlsx --output ./output \
  --amount-tolerance 0.05 --date-tolerance-days 5 --verbose
```

## 8 Matching Rules

| Rule | Narration | Date | Amount | Notes |
|------|-----------|------|--------|-------|
| R1 | Exact | Exact | Offset ±tol | Highest confidence (1.0) |
| R2 | Fuzzy ≥ threshold | Exact | Offset ±tol | Score = fuzzy_ratio/100 |
| R3 | Exact | Within day range | Offset ±tol | Score = 0.85 |
| R4 | Fuzzy ≥ threshold | Within day range | Offset ±tol | Score = 0.75 × (fuzzy/100) |
| R5 | Ignored | Exact | Offset ±tol | Score = 0.65 |
| R6 | Ignored | Ignored | Offset ±tol | Score = 0.50 |
| R7 | Ignored | Ignored | Offset ±tol | Same entity-pair reversals. Score = 0.70 |
| R8 | N/A | N/A | Group offset | Many-to-many flag. Flagged for review. |

## AI Touchpoints

1. **Anomaly Analysis** — Claude classifies unmatched transactions
2. **Complex Match Scoring** — Claude judges R8 many-to-many groups
3. **Executive Summary** — Claude writes a professional reconciliation summary

## Output Files

- `{run_id}_reconciliation.xlsx` — Multi-sheet Excel report
- `{run_id}_audit.json` — Machine-readable audit trail
