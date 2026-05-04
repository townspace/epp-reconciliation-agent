from pydantic import BaseModel
from enum import Enum
from datetime import datetime
from typing import Optional, List


class MatchRule(str, Enum):
    R1 = "R1_narr_exact_date_exact"
    R2 = "R2_narr_fuzzy_date_exact"
    R3 = "R3_narr_exact_date_range"
    R4 = "R4_narr_fuzzy_date_range"
    R5 = "R5_date_exact_only"
    R6 = "R6_amount_only"
    R7 = "R7_reversal_detection"
    R8 = "R8_many_to_many_flag"


class MatchResult(BaseModel):
    recon_id: str
    left_indices: List[int]
    right_indices: List[int]
    rule: MatchRule
    confidence_score: float
    ai_judgment: Optional[str] = None
    flagged_for_review: bool = False


class ValidationReport(BaseModel):
    missing_values: dict
    duplicates: List[int]
    currency_issues: List[dict]
    passed: bool


class AnomalyResult(BaseModel):
    transaction_index: int
    classification: str
    explanation: str
    suggested_action: str


class ComplexMatchJudgment(BaseModel):
    group_id: str
    verdict: str
    reasoning: str
    confidence: float


class ReconciliationReport(BaseModel):
    run_id: str
    run_timestamp: datetime
    total_transactions: int
    matched_pairs: List[MatchResult]
    unmatched_transactions: List[int]
    complex_groups: List[MatchResult]
    anomaly_analysis: List[AnomalyResult]
    complex_judgments: List[ComplexMatchJudgment]
    executive_summary: str
    match_rate_pct: float
