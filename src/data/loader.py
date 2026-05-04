import logging
import pandas as pd
from typing import Tuple

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = {"Entity", "PartnerEntity", "TransactionDate", "Amount", "Currency", "Narration"}

# Flexible column name mappings: canonical name → list of accepted aliases
COLUMN_ALIASES = {
    "Entity":          ["entity", "company", "company name", "legal entity", "account name", "account"],
    "PartnerEntity":   ["partnerentity", "partner entity", "counterparty", "counter party",
                        "bank", "bank name", "payee", "payer", "vendor", "customer"],
    "TransactionDate": ["transactiondate", "transaction date", "date", "value date",
                        "posting date", "entry date", "txn date", "posting dt"],
    "Amount":          ["amount", "value", "txn amount", "transaction amount", "net amount",
                        "debit-credit", "credit-debit"],
    "Currency":        ["currency", "ccy", "curr", "currency code"],
    "Narration":       ["narration", "narr", "description", "details", "particulars",
                        "reference", "ref", "memo", "remarks", "narrative"],
}


def _normalise_columns(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """Rename columns to canonical names using alias matching."""
    df = df.copy()
    rename_map = {}
    lower_cols = {c.lower().strip(): c for c in df.columns}
    for canonical, aliases in COLUMN_ALIASES.items():
        if canonical in df.columns:
            continue
        for alias in aliases:
            if alias in lower_cols:
                rename_map[lower_cols[alias]] = canonical
                break
    if rename_map:
        logger.info("'%s': renaming columns %s", label, rename_map)
        df = df.rename(columns=rename_map)
    return df


def _validate_columns(df: pd.DataFrame, label: str) -> None:
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"Dataset '{label}' is missing required column(s): {', '.join(sorted(missing))}"
        )


# ── Bank reconciliation specific loader ───────────────────────────────────────

def _prep_ledger(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map Ledger_InputData columns → canonical schema.

    Ledger columns:
      Posting Dt, Debit-Credit, Bank, Description, Particulars, ...
    """
    df = df.copy()
    df = df.dropna(how="all").reset_index(drop=True)

    # Date
    df["TransactionDate"] = pd.to_datetime(df.get("Posting Dt", df.get("Posting Date")), errors="coerce")

    # Amount — already signed (negative = payment out)
    df["Amount"] = pd.to_numeric(df.get("Debit-Credit", df.get("Amount")), errors="coerce")

    # Narration — combine Description + Particulars
    narr_parts = []
    for col in ["Description", "Particulars", "GL A/c Description"]:
        if col in df.columns:
            narr_parts.append(df[col].fillna("").astype(str).str.strip())
    df["Narration"] = narr_parts[0] if narr_parts else ""
    for part in narr_parts[1:]:
        df["Narration"] = df["Narration"].where(df["Narration"] != "", part)

    # Entity / PartnerEntity
    # Ledger = company books; Bank column = which bank account
    bank_col = df.get("Bank", df.get("BANK"))
    if bank_col is not None:
        df["PartnerEntity"] = bank_col.fillna("BANK").astype(str)
    else:
        df["PartnerEntity"] = "BANK"
    df["Entity"] = "COMPANY"

    # Currency — not in ledger, default USD
    df["Currency"] = df.get("Currency", "USD")

    df["dataset_source"] = "ledger"
    return df


def _prep_bank_statement(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map Statement_InputData columns → canonical schema.

    Bank Statement columns:
      Date, Credit-Debit, BANK, Narration, Narration 2, Credit, Debit, ...
    """
    df = df.copy()
    df = df.dropna(how="all").reset_index(drop=True)

    # Date
    df["TransactionDate"] = pd.to_datetime(df.get("Date", df.get("Value Date")), errors="coerce")

    # Amount — Credit-Debit is signed; if missing, compute from Credit/Debit columns
    if "Credit-Debit" in df.columns:
        df["Amount"] = pd.to_numeric(df["Credit-Debit"], errors="coerce")
    elif "Credit" in df.columns and "Debit" in df.columns:
        credit = pd.to_numeric(df["Credit"], errors="coerce").fillna(0)
        debit  = pd.to_numeric(df["Debit"],  errors="coerce").fillna(0)
        df["Amount"] = credit - debit   # positive = credit, negative = debit
    else:
        df["Amount"] = pd.to_numeric(df.get("Amount", 0), errors="coerce")

    # Flip sign so ledger and bank statement amounts offset each other
    # Ledger: payment out = negative; Bank: same payment = also negative
    # → negate bank so abs(ledger_amt + bank_amt) ≈ 0
    df["Amount"] = -df["Amount"]

    # Narration — combine Narration + Narration 2
    narr = df.get("Narration", pd.Series([""] * len(df))).fillna("").astype(str).str.strip()
    narr2 = df.get("Narration 2", pd.Series([""] * len(df))).fillna("").astype(str).str.strip()
    df["Narration"] = narr.where(narr != "", narr2)
    mask = (narr != "") & (narr2 != "")
    df.loc[mask, "Narration"] = narr[mask] + " " + narr2[mask]

    # Entity / PartnerEntity (mirror of ledger)
    bank_col = df.get("BANK", df.get("Bank"))
    if bank_col is not None:
        df["Entity"] = bank_col.fillna("BANK").astype(str)
    else:
        df["Entity"] = "BANK"
    df["PartnerEntity"] = "COMPANY"

    # Currency
    df["Currency"] = df.get("Currency", "USD")

    df["dataset_source"] = "bank_statement"
    return df


def _is_bank_recon_ledger(df: pd.DataFrame) -> bool:
    """Detect if file looks like the bank recon ledger format."""
    cols_lower = {c.lower().strip() for c in df.columns}
    return "debit-credit" in cols_lower or "posting dt" in cols_lower


def _is_bank_recon_statement(df: pd.DataFrame) -> bool:
    """Detect if file looks like the bank recon statement format."""
    cols_lower = {c.lower().strip() for c in df.columns}
    return "credit-debit" in cols_lower or ("credit" in cols_lower and "debit" in cols_lower)


def load_two_files(ledger_path: str, bank_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load ledger and bank statement from two separate files.
    Auto-detects bank reconciliation format vs generic intercompany format.
    """
    def _read(path):
        if str(path).lower().endswith(".csv"):
            return pd.read_csv(path)
        xl = pd.ExcelFile(path)
        return xl.parse(xl.sheet_names[0])

    raw_ledger = _read(ledger_path)
    raw_bank   = _read(bank_path)

    # Auto-detect bank reconciliation format
    if _is_bank_recon_ledger(raw_ledger) or _is_bank_recon_statement(raw_bank):
        logger.info("Detected bank reconciliation format — applying specialised column mapping")
        df1 = _prep_ledger(raw_ledger)
        df2 = _prep_bank_statement(raw_bank)
    else:
        # Generic intercompany format
        df1 = raw_ledger.copy()
        df2 = raw_bank.copy()
        df1 = _normalise_columns(df1, "ledger")
        df2 = _normalise_columns(df2, "bank_statement")
        df1 = df1.dropna(how="all").reset_index(drop=True)
        df2 = df2.dropna(how="all").reset_index(drop=True)
        _validate_columns(df1, "ledger")
        _validate_columns(df2, "bank_statement")
        df1["dataset_source"] = "ledger"
        df2["dataset_source"] = "bank_statement"

    logger.info("Loaded ledger: %d rows, bank_statement: %d rows", len(df1), len(df2))
    return df1, df2


# ── Single combined file loader ───────────────────────────────────────────────

def load_data(path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load from a single Excel file with two sheets or a blank-row separator."""
    xl = pd.ExcelFile(path)

    if len(xl.sheet_names) >= 2:
        df1 = xl.parse(xl.sheet_names[0])
        df2 = xl.parse(xl.sheet_names[1])
        logger.info("Loaded two sheets: %s, %s", xl.sheet_names[0], xl.sheet_names[1])
    else:
        raw = xl.parse(xl.sheet_names[0])
        blank_mask = raw.isnull().all(axis=1)
        blank_rows = blank_mask[blank_mask].index.tolist()
        if not blank_rows:
            raise ValueError(
                "Single-sheet file has no blank-row separator and only one sheet."
            )
        split_at = blank_rows[0]
        df1 = raw.iloc[:split_at].reset_index(drop=True)
        df2 = raw.iloc[split_at + 1:].reset_index(drop=True)
        if not df2.empty and all(isinstance(v, str) for v in df2.iloc[0]):
            df2.columns = df2.iloc[0]
            df2 = df2.iloc[1:].reset_index(drop=True)
        logger.info("Split single sheet at row %d into two datasets", split_at)

    df1 = _normalise_columns(df1, "dataset_1")
    df2 = _normalise_columns(df2, "dataset_2")
    _validate_columns(df1, "dataset_1")
    _validate_columns(df2, "dataset_2")
    df1["dataset_source"] = "dataset_1"
    df2["dataset_source"] = "dataset_2"

    logger.info("Loaded dataset_1: %d rows, dataset_2: %d rows", len(df1), len(df2))
    return df1, df2
