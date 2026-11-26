"""Aggregate classified quasi-sentences into a RILE-style left-right score.

Takes the output of `classify_manifestos.py` and returns one score per
manifesto, following MARPOR's RILE logic: the relative balance of right minus
left statements over all coded quasi-sentences.

    rile = (n_right - n_left) / n_total

`n_total` includes neutral sentences, as in MARPOR: a manifesto that spends
most of its length on non-positional content is correctly pulled towards zero.
The `--exclude-neutral` variant divides by (n_left + n_right) instead, which
measures the balance among positional statements only and is reported as a
sensitivity check.

Usage:
    python rile_score.py results/*.csv
    python rile_score.py results/*.csv --exclude-neutral --output rile_scores.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def rile(labels: pd.Series, exclude_neutral: bool = False) -> float:
    """Return the RILE-style score for one manifesto's labels."""
    counts = labels.str.strip().str.lower().value_counts()
    n_left = int(counts.get("left", 0))
    n_right = int(counts.get("right", 0))
    n_neutral = int(counts.get("neutral", 0))

    denominator = (n_left + n_right) if exclude_neutral else (n_left + n_right + n_neutral)
    if denominator == 0:
        return float("nan")
    return (n_right - n_left) / denominator


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("files", nargs="+", type=Path, help="classified CSV files, one per manifesto")
    parser.add_argument("--exclude-neutral", action="store_true", help="divide by positional sentences only")
    parser.add_argument("--output", type=Path, help="write the table to this CSV instead of stdout")
    args = parser.parse_args()

    rows = []
    for path in args.files:
        df = pd.read_csv(path)
        labelled = df["orientation"].dropna().astype(str)
        labelled = labelled[labelled != ""]
        rows.append(
            {
                # File names follow the thesis shorthand [Party][ISO][Year],
                # e.g. SPDGER2013 -> SPD, Germany, 2013 election.
                "manifesto": path.stem.replace("Reasoning", ""),
                "n_coded": len(labelled),
                "n_missing": int(len(df) - len(labelled)),
                "rile": round(rile(labelled, args.exclude_neutral), 4),
            }
        )

    table = pd.DataFrame(rows).sort_values("manifesto")
    if args.output:
        table.to_csv(args.output, index=False)
        print("Written:", args.output)
    else:
        print(table.to_string(index=False))


if __name__ == "__main__":
    main()
