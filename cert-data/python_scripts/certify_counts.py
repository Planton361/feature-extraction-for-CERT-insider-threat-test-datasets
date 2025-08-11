#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compute counts to substantiate claims for CERT r5.2:

- Total activities: sum of rows in the event logs (logon.csv, device.csv, http.csv, email.csv, file.csv)
- Malicious actions + insider users: by default from an external labels file (e.g., "dayr5.2.csv")
  with columns ["user", "insider"] where "insider" > 0 flags a positive entry.

IMPORTANT
---------
1) "Activities" here means *rows* in the primary event CSVs.
2) The interpretation of "malicious actions" depends on your labels file:
   - If your labels are per-event, summing the positive rows gives event-level malicious actions.
   - If your labels are per-day or per-session aggregates (common in some pipelines), the sum will
     NOT equal event-level actions. Document which definition you use in your thesis.
3) If you have an official per-event ground-truth (e.g., an events-level label mapping), point --labels
   to that file and adapt the "is_positive" logic below.

Usage
-----
python certify_counts.py --base /path/to/r5.2/ExtractedData --labels /path/to/dayr5.2.csv

Outputs a small summary table and prints totals.
"""
import argparse
import sys
import pandas as pd
from pathlib import Path

EVENT_FILES = ["logon.csv", "device.csv", "http.csv", "email.csv", "file.csv"]

def count_rows_csv(path: Path) -> int:
    # Efficient row count without loading entire DataFrame into memory
    # We count non-header lines
    n = 0
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        # skip header
        header = next(f, None)
        for _ in f:
            n += 1
    return n

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=str, required=True,
                    help="Directory containing CERT r5.2 event CSVs (logon.csv, device.csv, http.csv, email.csv, file.csv)")
    ap.add_argument("--labels", type=str, required=False,
                    help="Path to labels CSV (e.g., dayr5.2.csv) with columns ['user', 'insider'] or similar")
    ap.add_argument("--labels-user-col", type=str, default="user",
                    help="Column name in labels file for user id (default: 'user')")
    ap.add_argument("--labels-flag-col", type=str, default="insider",
                    help="Column name in labels file for positive flag (default: 'insider')")
    ap.add_argument("--labels-positive-threshold", type=float, default=0.0,
                    help="Values strictly greater than this are treated as positive (default: 0.0)")
    args = ap.parse_args()

    base = Path(args.base)
    if not base.exists():
        print(f"[ERROR] Base directory not found: {base}", file=sys.stderr)
        sys.exit(1)

    # (A) Total activities = sum of rows across event logs
    activities_by_file = {}
    total_activities = 0
    for fname in EVENT_FILES:
        fpath = base / fname
        if not fpath.exists():
            print(f"[WARN] Missing file: {fpath} (count as 0)")
            activities_by_file[fname] = 0
            continue
        nrows = count_rows_csv(fpath)
        activities_by_file[fname] = nrows
        total_activities += nrows

    # (B) Malicious actions + insider users (from labels)
    malicious_actions = None
    insider_users = None
    label_granularity = "unknown"
    if args.labels:
        lpath = Path(args.labels)
        if not lpath.exists():
            print(f"[ERROR] Labels file not found: {lpath}", file=sys.stderr)
            sys.exit(2)
        # Load minimal columns
        usecols = [args.labels_user_col, args.labels_flag_col]
        df_labels = pd.read_csv(lpath, usecols=usecols)
        # Positive if > threshold
        pos_mask = df_labels[args.labels_flag_col] > args.labels_positive_threshold
        malicious_actions = int(pos_mask.sum())
        insider_users = int(df_labels.loc[pos_mask, args.labels_user_col].nunique())

        # Heuristic note about granularity
        label_granularity = "event-or-aggregate (depends on labels)"
    else:
        print("[INFO] No labels file provided. Malicious counts will be left empty.")

    # (C) Summary
    summary = {
        "logon.csv": activities_by_file.get("logon.csv", 0),
        "device.csv": activities_by_file.get("device.csv", 0),
        "http.csv": activities_by_file.get("http.csv", 0),
        "email.csv": activities_by_file.get("email.csv", 0),
        "file.csv": activities_by_file.get("file.csv", 0),
        "TOTAL_ACTIVITIES": total_activities,
        "MALICIOUS_ACTIONS": malicious_actions,
        "INSIDER_USERS": insider_users,
        "LABELS_GRANULARITY_NOTE": label_granularity,
        "INSIDER_RATE_percent": (malicious_actions / total_activities * 100.0) if (malicious_actions is not None and total_activities > 0) else None,
    }
    out = pd.Series(summary)
    out_path = Path("cert_r52_activity_summary.csv")
    out.to_csv(out_path, header=False)
    print("\n=== CERT r5.2 Summary ===")
    for k, v in summary.items():
        print(f"{k:>24}: {v}")
    print(f"\n[Saved] {out_path.resolve()}")

if __name__ == "__main__":
    main()
