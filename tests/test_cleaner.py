import pytest
import pandas as pd
import numpy as np
from src.data.cleaner import normalize_amounts, normalize_dates, normalize_narration


def make_df(**kwargs):
    return pd.DataFrame({k: [v] for k, v in kwargs.items()})


class TestNormalizeAmounts:
    def test_plain_float(self):
        df = make_df(Amount=123.45)
        result = normalize_amounts(df)
        assert result["Amount"].iloc[0] == pytest.approx(123.45)

    def test_strips_currency_symbols(self):
        df = make_df(Amount="₹1,234.56")
        result = normalize_amounts(df)
        assert result["Amount"].iloc[0] == pytest.approx(1234.56)

    def test_negative_amount(self):
        df = make_df(Amount="-500.00")
        result = normalize_amounts(df)
        assert result["Amount"].iloc[0] == pytest.approx(-500.0)

    def test_bad_value_becomes_nan(self):
        df = make_df(Amount="not_a_number")
        result = normalize_amounts(df)
        assert np.isnan(result["Amount"].iloc[0])

    def test_nan_stays_nan(self):
        df = make_df(Amount=np.nan)
        result = normalize_amounts(df)
        assert np.isnan(result["Amount"].iloc[0])

    def test_dollar_sign(self):
        df = make_df(Amount="$9,999.00")
        result = normalize_amounts(df)
        assert result["Amount"].iloc[0] == pytest.approx(9999.0)


class TestNormalizeDates:
    def test_iso_date_string(self):
        df = make_df(TransactionDate="2024-01-15")
        result = normalize_dates(df)
        assert result["TransactionDate"].iloc[0] == pd.Timestamp("2024-01-15")

    def test_datetime_object(self):
        df = make_df(TransactionDate=pd.Timestamp("2024-06-01"))
        result = normalize_dates(df)
        assert result["TransactionDate"].iloc[0] == pd.Timestamp("2024-06-01")

    def test_bad_date_becomes_nat(self):
        df = make_df(TransactionDate="not-a-date")
        result = normalize_dates(df)
        assert pd.isna(result["TransactionDate"].iloc[0])

    def test_nan_stays_nat(self):
        df = make_df(TransactionDate=np.nan)
        result = normalize_dates(df)
        assert pd.isna(result["TransactionDate"].iloc[0])


class TestNormalizeNarration:
    def test_single_narration_column(self):
        df = make_df(Narration="Payment for Invoice 123")
        result = normalize_narration(df)
        assert result["narration_clean"].iloc[0] == "payment for invoice 123"

    def test_strips_extra_whitespace(self):
        df = make_df(Narration="  multiple   spaces  ")
        result = normalize_narration(df)
        assert result["narration_clean"].iloc[0] == "multiple spaces"

    def test_multiple_narration_columns(self):
        df = pd.DataFrame({"Narration": ["Part A"], "Narration2": ["Part B"]})
        result = normalize_narration(df)
        assert result["narration_clean"].iloc[0] == "part a | part b"

    def test_null_narration_handled(self):
        df = pd.DataFrame({"Narration": [None]})
        result = normalize_narration(df)
        assert result["narration_clean"].iloc[0] == ""
