"""Shared pytest fixtures for CausalSynth tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from causalsynth.data.schema import CausalDAG

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def fixture_dag_path() -> Path:
    return FIXTURES_DIR / "health_dag.json"


@pytest.fixture
def fixture_data_path() -> Path:
    return FIXTURES_DIR / "real_health_data.csv"


@pytest.fixture
def simple_dag() -> CausalDAG:
    """A simple 3-node DAG: age → bmi → blood_pressure."""
    return CausalDAG(
        nodes=["age", "bmi", "blood_pressure"],
        edges=[("age", "bmi"), ("age", "blood_pressure"), ("bmi", "blood_pressure")],
    )


@pytest.fixture
def health_dag() -> CausalDAG:
    """A health-domain DAG with 4 nodes."""
    return CausalDAG(
        nodes=["age", "bmi", "exercise_hours", "blood_pressure"],
        edges=[
            ("age", "bmi"),
            ("age", "blood_pressure"),
            ("bmi", "blood_pressure"),
            ("exercise_hours", "bmi"),
            ("exercise_hours", "blood_pressure"),
        ],
    )


@pytest.fixture
def health_df(health_dag: CausalDAG) -> pd.DataFrame:
    """Synthetic health dataset consistent with health_dag."""
    rng = np.random.default_rng(42)
    n = 200
    age = rng.normal(45, 10, n)
    exercise = rng.normal(3, 1.5, n).clip(0, 10)
    bmi = 18 + 0.1 * age - 0.5 * exercise + rng.normal(0, 2, n)
    bp = 90 + 0.3 * age + 0.8 * bmi - 1.2 * exercise + rng.normal(0, 5, n)

    return pd.DataFrame({
        "age": age,
        "bmi": bmi.clip(15, 45),
        "exercise_hours": exercise,
        "blood_pressure": bp.clip(70, 180),
    })
