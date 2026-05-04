"""Shared pytest fixtures."""
import pytest
from pathlib import Path

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def sample_input_path():
    p = FIXTURES_DIR / "sample_input.xlsx"
    assert p.exists(), f"Fixture not found: {p}. Run: python scripts/generate_test_data.py"
    return str(p)
