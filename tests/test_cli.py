"""CLI tests for CausalSynth."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from causalsynth.cli import app

runner = CliRunner()


class TestGenerateCommand:
    def test_generate_with_fixture(
        self, fixture_data_path: Path, fixture_dag_path: Path, tmp_path: Path
    ) -> None:
        """generate command produces a synthetic CSV."""
        output = tmp_path / "synth.csv"
        result = runner.invoke(
            app,
            [
                "generate",
                str(fixture_data_path),
                str(fixture_dag_path),
                str(output),
                "--samples", "50",
                "--no-validate",
            ],
        )
        assert result.exit_code == 0
        assert output.exists()

    def test_generate_creates_output_with_correct_rows(
        self, fixture_data_path: Path, fixture_dag_path: Path, tmp_path: Path
    ) -> None:
        """Output CSV has the requested number of rows."""
        import pandas as pd

        output = tmp_path / "synth.csv"
        result = runner.invoke(
            app,
            [
                "generate",
                str(fixture_data_path),
                str(fixture_dag_path),
                str(output),
                "--samples", "75",
                "--no-validate",
            ],
        )
        assert result.exit_code == 0
        df = pd.read_csv(output)
        assert len(df) == 75

    def test_generate_missing_data_exits_nonzero(
        self, fixture_dag_path: Path, tmp_path: Path
    ) -> None:
        result = runner.invoke(
            app,
            ["generate", str(tmp_path / "nope.csv"), str(fixture_dag_path), str(tmp_path / "out.csv")],
        )
        assert result.exit_code != 0

    def test_generate_with_validation(
        self, fixture_data_path: Path, fixture_dag_path: Path, tmp_path: Path
    ) -> None:
        """generate with --validate runs and shows fidelity."""
        output = tmp_path / "synth.csv"
        result = runner.invoke(
            app,
            [
                "generate",
                str(fixture_data_path),
                str(fixture_dag_path),
                str(output),
                "--samples", "50",
                "--validate",
            ],
        )
        assert result.exit_code == 0
        assert "fidelity" in result.output.lower() or "Validation" in result.output


class TestValidateCommand:
    def test_validate_against_itself(
        self, fixture_data_path: Path, fixture_dag_path: Path
    ) -> None:
        """Validating a dataset against itself should show high fidelity."""
        result = runner.invoke(
            app,
            [
                "validate",
                str(fixture_data_path),
                str(fixture_data_path),
                str(fixture_dag_path),
            ],
        )
        assert result.exit_code == 0
        assert "fidelity" in result.output.lower() or "PASSED" in result.output
