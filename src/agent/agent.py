"""
AI agent — called at exactly three decision points.
Uses OpenAI API (gpt-4o). Retry with exponential backoff (max 3 attempts).
"""
import json
import logging
import time
from typing import List

from openai import OpenAI, RateLimitError, APIError

from src.agent.prompts import (
    SYSTEM_PROMPT_ANOMALY,
    SYSTEM_PROMPT_COMPLEX_MATCH,
    SYSTEM_PROMPT_SUMMARY,
)
from src.config import Config
from src.models import AnomalyResult, ComplexMatchJudgment, MatchResult, ReconciliationReport

logger = logging.getLogger(__name__)

MODEL = "gpt-4o"
BATCH_SIZE = 20


# ── retry helper ─────────────────────────────────────────────────────────────

def _call_with_retry(client: OpenAI, **kwargs) -> str:
    """Call chat.completions.create and return the text content."""
    last_exc = None
    for attempt in range(3):
        try:
            response = client.chat.completions.create(**kwargs)
            return response.choices[0].message.content.strip()
        except RateLimitError as e:
            wait = 2 ** attempt
            logger.warning("Rate limit hit (attempt %d/3); sleeping %ds", attempt + 1, wait)
            time.sleep(wait)
            last_exc = e
        except APIError as e:
            if attempt == 2:
                raise
            logger.warning("API error (attempt %d/3): %s", attempt + 1, e)
            time.sleep(1)
            last_exc = e
    raise RuntimeError(f"Max retries exceeded: {last_exc}")


# ── AI touchpoint 1: anomaly analysis ────────────────────────────────────────

def analyze_unmatched(transactions: list, config: Config) -> List[AnomalyResult]:
    if config.skip_ai:
        logger.info("skip_ai=True; skipping anomaly analysis")
        return []

    client = OpenAI(api_key=config.openai_api_key)
    results: List[AnomalyResult] = []

    for batch_start in range(0, len(transactions), BATCH_SIZE):
        batch = transactions[batch_start: batch_start + BATCH_SIZE]
        user_msg = json.dumps(batch, default=str)

        raw = _call_with_retry(
            client,
            model=MODEL,
            max_tokens=2048,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT_ANOMALY},
                {"role": "user", "content": user_msg},
            ],
        )
        try:
            items = json.loads(raw)
            for item in items:
                results.append(AnomalyResult(**item))
        except (json.JSONDecodeError, TypeError, ValueError) as e:
            logger.error("Failed to parse anomaly response: %s\nRaw: %s", e, raw[:500])

    logger.info("Anomaly analysis complete: %d classifications", len(results))
    return results


# ── AI touchpoint 2: complex match scoring ────────────────────────────────────

def score_complex_match(group: MatchResult, df, config: Config) -> ComplexMatchJudgment:
    if config.skip_ai:
        return ComplexMatchJudgment(
            group_id=group.recon_id,
            verdict="NEEDS_REVIEW",
            reasoning="AI scoring skipped (skip_ai=True).",
            confidence=0.0,
        )

    client = OpenAI(api_key=config.openai_api_key)

    left_rows = df.loc[group.left_indices].to_dict(orient="records")
    right_rows = df.loc[group.right_indices].to_dict(orient="records")
    total_left = sum(r.get("Amount", 0) for r in left_rows if r.get("Amount") is not None)
    total_right = sum(r.get("Amount", 0) for r in right_rows if r.get("Amount") is not None)
    narrations = list({str(r.get("narration_clean", "")) for r in left_rows + right_rows if r.get("narration_clean")})

    payload = {
        "group_id": group.recon_id,
        "left_transactions": left_rows,
        "right_transactions": right_rows,
        "total_left": total_left,
        "total_right": total_right,
        "narrations": narrations,
    }

    raw = _call_with_retry(
        client,
        model=MODEL,
        max_tokens=512,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT_COMPLEX_MATCH},
            {"role": "user", "content": json.dumps(payload, default=str)},
        ],
    )
    try:
        data = json.loads(raw)
        return ComplexMatchJudgment(**data)
    except (json.JSONDecodeError, TypeError, ValueError) as e:
        logger.error("Failed to parse complex match response: %s\nRaw: %s", e, raw[:500])
        return ComplexMatchJudgment(
            group_id=group.recon_id,
            verdict="NEEDS_REVIEW",
            reasoning="Parse error in AI response.",
            confidence=0.0,
        )


# ── AI touchpoint 3: executive summary ───────────────────────────────────────

def generate_summary(report: ReconciliationReport, config: Config) -> str:
    if config.skip_ai:
        return _fallback_summary(report)

    client = OpenAI(api_key=config.openai_api_key)

    from collections import Counter
    rule_counts = Counter(m.rule.value for m in report.matched_pairs)
    anomaly_counts = Counter(a.classification for a in report.anomaly_analysis)

    stats = (
        f"Run ID: {report.run_id}\n"
        f"Total transactions: {report.total_transactions}\n"
        f"Overall match rate: {report.match_rate_pct:.1f}%\n"
        f"Matched pairs: {len(report.matched_pairs)}\n"
        f"Complex groups (R8): {len(report.complex_groups)}\n"
        f"Unmatched transactions: {len(report.unmatched_transactions)}\n\n"
        f"Matches by rule:\n"
        + "\n".join(f"  {rule}: {count}" for rule, count in sorted(rule_counts.items()))
        + f"\n\nTop anomaly classifications:\n"
        + "\n".join(f"  {cls}: {cnt}" for cls, cnt in anomaly_counts.most_common(5))
    )

    summary = _call_with_retry(
        client,
        model=MODEL,
        max_tokens=600,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT_SUMMARY},
            {"role": "user", "content": stats},
        ],
    )
    logger.info("Executive summary generated (%d chars)", len(summary))
    return summary


def _fallback_summary(report: ReconciliationReport) -> str:
    from collections import Counter
    rule_counts = Counter(m.rule.value for m in report.matched_pairs)
    lines = [
        f"Reconciliation Run: {report.run_id}",
        f"Total transactions: {report.total_transactions}",
        f"Match rate: {report.match_rate_pct:.1f}%",
        f"Matched pairs: {len(report.matched_pairs)}",
        f"Complex groups (R8): {len(report.complex_groups)}",
        f"Unmatched: {len(report.unmatched_transactions)}",
        "",
        "Rule breakdown:",
    ] + [f"  {r}: {c}" for r, c in sorted(rule_counts.items())]
    return "\n".join(lines)
