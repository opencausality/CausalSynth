"""Tests for the SCM builder."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from causalsynth.data.schema import CausalDAG, SCM
from causalsynth.exceptions import SCMFitError
from causalsynth.scm.builder import fit_scm


class TestFitSCM:
    def test_returns_scm_instance(self, health_df: pd.DataFrame, health_dag: CausalDAG) -> None:
        """fit_scm returns an SCM instance."""
        scm = fit_scm(health_df, health_dag)
        assert isinstance(scm, SCM)

    def test_equations_count_matches_numeric_nodes(
        self, health_df: pd.DataFrame, health_dag: CausalDAG
    ) -> None:
        """One structural equation per numeric node."""
        scm = fit_scm(health_df, health_dag)
        assert len(scm.equations) == len(health_dag.nodes)

    def test_topological_order_respects_dag(
        self, health_df: pd.DataFrame, health_dag: CausalDAG
    ) -> None:
        """All parents appear before their children in topological order."""
        scm = fit_scm(health_df, health_dag)
        order = scm.topological_order
        for cause, effect in health_dag.edges:
            if cause in order and effect in order:
                assert order.index(cause) < order.index(effect), (
                    f"'{cause}' should come before '{effect}' in topological order"
                )

    def test_root_nodes_have_no_parents(
        self, health_df: pd.DataFrame, health_dag: CausalDAG
    ) -> None:
        """Root nodes have no parents in their structural equation."""
        scm = fit_scm(health_df, health_dag)
        # age and exercise_hours are root nodes
        eq_map = {eq.variable: eq for eq in scm.equations}
        if "age" in eq_map:
            assert eq_map["age"].parents == []
        if "exercise_hours" in eq_map:
            assert eq_map["exercise_hours"].parents == []

    def test_feature_stats_populated(
        self, health_df: pd.DataFrame, health_dag: CausalDAG
    ) -> None:
        """feature_stats contains mean/std for each node."""
        scm = fit_scm(health_df, health_dag)
        for node in health_dag.nodes:
            assert node in scm.feature_stats
            assert "mean" in scm.feature_stats[node]
            assert "std" in scm.feature_stats[node]

    def test_cyclic_dag_raises_error(self, health_df: pd.DataFrame) -> None:
        """Cyclic DAG raises validation error."""
        from pydantic import ValidationError

        with pytest.raises((ValidationError, Exception)):
            CausalDAG(
                nodes=["a", "b"],
                edges=[("a", "b"), ("b", "a")],  # cycle
            )

    def test_all_gaussian_noise_type(
        self, health_df: pd.DataFrame, health_dag: CausalDAG
    ) -> None:
        """All equations use gaussian noise when specified."""
        scm = fit_scm(health_df, health_dag, noise_type="gaussian")
        for eq in scm.equations:
            assert eq.noise_type == "gaussian"
