"""Tests for the privacy noise module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from causalsynth.generation.privacy import apply_privacy_noise


class TestApplyPrivacyNoise:
    def test_returns_dataframe(self) -> None:
        """apply_privacy_noise returns a DataFrame."""
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
        result = apply_privacy_noise(df, epsilon=1.0, seed=42)
        assert isinstance(result, pd.DataFrame)

    def test_same_shape_as_input(self) -> None:
        """Output has the same shape as input."""
        df = pd.DataFrame(np.random.default_rng(0).standard_normal((50, 4)))
        result = apply_privacy_noise(df, epsilon=1.0, seed=42)
        assert result.shape == df.shape

    def test_output_differs_from_input(self) -> None:
        """Privacy noise actually changes the values."""
        df = pd.DataFrame({"x": [1.0] * 20, "y": [2.0] * 20})
        result = apply_privacy_noise(df, epsilon=1.0, seed=42)
        # At least some values should differ
        assert not result.equals(df)

    def test_reproducible_with_same_seed(self) -> None:
        """Same seed produces identical noisy output."""
        df = pd.DataFrame({"x": np.arange(10, dtype=float)})
        r1 = apply_privacy_noise(df, epsilon=1.0, seed=7)
        r2 = apply_privacy_noise(df, epsilon=1.0, seed=7)
        pd.testing.assert_frame_equal(r1, r2)

    def test_larger_epsilon_less_noise(self) -> None:
        """Larger epsilon (less privacy) should add less noise on average."""
        rng = np.random.default_rng(0)
        df = pd.DataFrame({"x": rng.standard_normal(1000)})

        r_tight = apply_privacy_noise(df.copy(), epsilon=0.1, seed=42)
        r_loose = apply_privacy_noise(df.copy(), epsilon=10.0, seed=42)

        tight_diff = (r_tight["x"] - df["x"]).abs().mean()
        loose_diff = (r_loose["x"] - df["x"]).abs().mean()

        assert tight_diff > loose_diff, (
            f"Expected tighter privacy (ε=0.1) to add more noise: {tight_diff:.4f} vs {loose_diff:.4f}"
        )

    def test_column_names_preserved(self) -> None:
        """Column names are unchanged after privacy noise."""
        df = pd.DataFrame({"age": [25.0, 30.0], "bmi": [22.0, 26.0]})
        result = apply_privacy_noise(df, epsilon=1.0, seed=42)
        assert list(result.columns) == list(df.columns)
