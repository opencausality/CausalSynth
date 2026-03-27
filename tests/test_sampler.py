"""Tests for the ancestral sampler."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from causalsynth.data.schema import CausalDAG
from causalsynth.generation.sampler import generate_samples
from causalsynth.scm.builder import fit_scm


class TestGenerateSamples:
    def test_returns_dataframe(self, health_df: pd.DataFrame, health_dag: CausalDAG) -> None:
        """generate_samples returns a pandas DataFrame."""
        scm = fit_scm(health_df, health_dag)
        synth = generate_samples(scm, n_samples=100, seed=42)
        assert isinstance(synth, pd.DataFrame)

    def test_correct_row_count(self, health_df: pd.DataFrame, health_dag: CausalDAG) -> None:
        """Returned DataFrame has exactly n_samples rows."""
        scm = fit_scm(health_df, health_dag)
        synth = generate_samples(scm, n_samples=250, seed=42)
        assert len(synth) == 250

    def test_columns_match_scm_variables(
        self, health_df: pd.DataFrame, health_dag: CausalDAG
    ) -> None:
        """Synthetic DataFrame columns match SCM topological order."""
        scm = fit_scm(health_df, health_dag)
        synth = generate_samples(scm, n_samples=100, seed=42)
        for var in scm.topological_order:
            assert var in synth.columns

    def test_reproducible_with_same_seed(
        self, health_df: pd.DataFrame, health_dag: CausalDAG
    ) -> None:
        """Same seed produces identical output."""
        scm = fit_scm(health_df, health_dag)
        synth1 = generate_samples(scm, n_samples=100, seed=99)
        synth2 = generate_samples(scm, n_samples=100, seed=99)
        pd.testing.assert_frame_equal(synth1, synth2)

    def test_different_seeds_produce_different_output(
        self, health_df: pd.DataFrame, health_dag: CausalDAG
    ) -> None:
        """Different seeds produce different output."""
        scm = fit_scm(health_df, health_dag)
        synth1 = generate_samples(scm, n_samples=100, seed=1)
        synth2 = generate_samples(scm, n_samples=100, seed=2)
        assert not synth1.equals(synth2)

    def test_zero_samples_raises(self, health_df: pd.DataFrame, health_dag: CausalDAG) -> None:
        """n_samples=0 raises ValueError."""
        scm = fit_scm(health_df, health_dag)
        with pytest.raises(ValueError, match="n_samples"):
            generate_samples(scm, n_samples=0)

    def test_no_nan_in_output(self, health_df: pd.DataFrame, health_dag: CausalDAG) -> None:
        """Synthetic data has no NaN values."""
        scm = fit_scm(health_df, health_dag)
        synth = generate_samples(scm, n_samples=200, seed=42)
        assert not synth.isnull().any().any()
