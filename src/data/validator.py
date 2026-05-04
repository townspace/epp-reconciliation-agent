import logging
import pandas as pd
from src.models import ValidationReport

logger = logging.getLogger(__name__)

_DUP_KEYS = ["Entity", "PartnerEntity", "TransactionDate", "Amount"]


def validate(df: pd.DataFrame) -> ValidationReport:
    # Missing values per column
    missing = {col: int(df[col].isna().sum()) for col in df.columns if df[col].isna().any()}

    # Duplicate transactions
    dup_mask = df.duplicated(subset=_DUP_KEYS, keep=False)
    duplicates = df.index[dup_mask].tolist()
    if duplicates:
        logger.warning("Found %d duplicate row(s) by key columns", len(duplicates))

    # Currency inconsistencies: same entity-pair + date + amount but different currencies
    currency_issues = []
    key_cols = ["Entity", "PartnerEntity", "TransactionDate", "Amount"]
    if all(c in df.columns for c in key_cols + ["Currency"]):
        grouped = df.groupby(key_cols)["Currency"].nunique()
        conflict_keys = grouped[grouped > 1].index
        for key in conflict_keys:
            mask = (
                (df["Entity"] == key[0])
                & (df["PartnerEntity"] == key[1])
                & (df["TransactionDate"] == key[2])
                & (df["Amount"] == key[3])
            )
            currencies = df.loc[mask, "Currency"].unique().tolist()
            currency_issues.append({
                "entity": key[0],
                "partner_entity": key[1],
                "date": str(key[2]),
                "amount": key[3],
                "currencies": currencies,
            })
            logger.warning("Currency mismatch for %s <-> %s: %s", key[0], key[1], currencies)

    passed = len(duplicates) == 0 and len(currency_issues) == 0
    return ValidationReport(
        missing_values=missing,
        duplicates=duplicates,
        currency_issues=currency_issues,
        passed=passed,
    )
