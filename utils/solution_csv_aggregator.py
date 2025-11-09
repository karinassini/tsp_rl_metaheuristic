from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


def _collect_csv_files(root: Path) -> Iterable[Path]:
    """Yield every CSV file found under the given root directory."""
    return (path for path in sorted(root.rglob("*.csv")) if path.is_file())


def concatenate_solution_csvs(
    solutions_root: str | Path,
    output_csv: str | Path | None = None,
    add_source_column: bool = True,
) -> pd.DataFrame:
    """Concatenate every CSV inside the solutions directory tree into one file.

    Parameters
    ----------
    solutions_root:
        Directory that contains per-instance solution CSV files. The search is
        recursive.
    output_csv:
        Destination file for the concatenated data. When ``None`` the file is
        written as ``solutions_combined.csv`` in ``solutions_root``.
    add_source_column:
        When ``True`` a ``source_file`` column is added with the CSV path
        relative to ``solutions_root``.

    Returns
    -------
    pandas.DataFrame
        The concatenated dataframe. Also written to ``output_csv``.

    Raises
    ------
    FileNotFoundError
        If ``solutions_root`` does not exist.
    ValueError
        If no CSV files are discovered within ``solutions_root``.
    RuntimeError
        If any CSV fails to load.
    """

    root_path = Path(solutions_root).expanduser().resolve()
    if not root_path.exists():
        raise FileNotFoundError(f"Solutions directory not found: {root_path}")

    csv_paths = list(_collect_csv_files(root_path))
    if not csv_paths:
        raise ValueError(f"No CSV files found under {root_path}")

    frames: list[pd.DataFrame] = []
    for csv_path in csv_paths:
        try:
            frame = pd.read_csv(csv_path)
        except Exception as exc:  # pragma: no cover - defensive guard
            raise RuntimeError(f"Failed to read CSV: {csv_path}") from exc
        if add_source_column:
            relative_path = csv_path.relative_to(root_path)
            frame.insert(0, "source_file", str(relative_path))
        frames.append(frame)

    combined = pd.concat(frames, ignore_index=True)

    destination = Path(output_csv).expanduser() if output_csv else root_path / "solutions_combined.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(destination, index=False)

    return combined

if __name__ == "__main__":
    import argparse

    default_root = Path("./outputs/solutions")

    concatenate_solution_csvs(
        solutions_root=default_root,
        output_csv=Path("./outputs/solutions_combined.csv"),
        add_source_column=None,
    )