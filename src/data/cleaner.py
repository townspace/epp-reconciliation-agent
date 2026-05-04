import logging
import re
import pandas as pd

logger = logging.getLogger(__name__)

_CURRENCY_SYMBOLS = re.compile(r"[₹$€£,\s]")


def normalize_amounts(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    def _parse(val):
        if pd.isna(val):
            return float("nan")
        cleaned = _CURRENCY_SYMBOLS.sub("", str(val)).strip()
        try:
            return float(cleaned)
        except ValueError:
            logger.warning("Could not parse amount value: %r — setting to NaN", val)
            return float("nan")

    df["Amount"] = df["Amount"].apply(_parse)
    return df


def normalize_dates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    def _parse(val):
        if pd.isna(val):
            return pd.NaT
        try:
            return pd.to_datetime(val)
        except Exception:
            logger.warning("Could not parse date value: %r — setting to NaT", val)
            return pd.NaT

    df["TransactionDate"] = df["TransactionDate"].apply(_parse)
    df["TransactionDate"] = pd.to_datetime(df["TransactionDate"], errors="coerce")
    return df


def normalize_narration(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    narr_cols = [c for c in df.columns if re.match(r"narr(ation)?", c, re.IGNORECASE)]

    def _combine(row):
        parts = [str(row[c]).strip() for c in narr_cols if not pd.isna(row[c]) and str(row[c]).strip()]
        combined = " | ".join(parts)
        combined = combined.lower()
        combined = re.sub(r"\s+", " ", combined).strip()
        return combined

    df["narration_clean"] = df.apply(_combine, axis=1)
    return df
