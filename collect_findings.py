#!/usr/bin/env python3
"""
Fill the `your_value` column of milestone1_findings_TEMPLATE.csv automatically
from the numbers the notebook saved in milestone1_results.json.

Usage (after running the notebook):
    ~/nutriscan-env/bin/python collect_findings.py
    ~/nutriscan-env/bin/python collect_findings.py --results path/to/milestone1_results.json

Writes milestone1_findings_FILLED.csv. Rows whose value cannot be filled
automatically are left blank for you - those are the ones that need your eyes
and your judgement, and they are the ones that earn the interpretation marks.
"""
import argparse, csv, json, sys
from pathlib import Path

HERE = Path(__file__).parent

# How to format each key when writing it into the CSV.
PERCENT_KEYS = {"detection_rate", "fallback_rate", "manual_precision_good",
                "manual_precision_good_or_partial", "median_box_area_frac"}
ALREADY_PCT  = {"pct_dark", "pct_bright", "pct_low_contrast", "pct_blurry",
                "pct_multi_region"}


def fmt(key, value):
    if value is None:
        return ""
    if isinstance(value, float):
        if key in PERCENT_KEYS:
            return f"{value * 100:.1f}%"
        if key in ALREADY_PCT:
            return f"{value:.1f}%"
        return f"{value:.3f}".rstrip("0").rstrip(".")
    return str(value)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=None,
                    help="path to milestone1_results.json (default: search nearby)")
    ap.add_argument("--template", default=str(HERE / "milestone1_findings_TEMPLATE.csv"))
    ap.add_argument("--out", default=str(HERE / "milestone1_findings_FILLED.csv"))
    args = ap.parse_args()

    # Find the results file.
    if args.results:
        results_path = Path(args.results)
    else:
        candidates = list(HERE.rglob("milestone1_results.json"))
        if not candidates:
            sys.exit(
                "Could not find milestone1_results.json.\n"
                "Run the notebook first (it is written by the 'Save outputs' cell),\n"
                "or pass the path with --results. On Colab, download it from\n"
                "MyDrive/NutriScan_M1/ first."
            )
        results_path = candidates[0]

    R = json.loads(results_path.read_text())
    print(f"Loaded {len(R)} values from {results_path}")

    with open(args.template, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    filled = manual = 0
    for row in rows:
        key = (row.get("R_key") or "").strip()
        if key and key != "-" and key in R:
            row["your_value"] = fmt(key, R[key])
            filled += 1
        elif not row.get("your_value"):
            manual += 1

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    print(f"Wrote {args.out}")
    print(f"  {filled} values filled automatically")
    print(f"  {manual} rows still need you (read them from the notebook, or judge them yourself)")
    print()
    print("Next: open the CSV and write the 'what_it_means_write_one_sentence' column.")
    print("That column is where the interpretation marks are.")


if __name__ == "__main__":
    main()
