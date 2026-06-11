from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path
from typing import Any, Iterator

import pandas as pd

VALID_LABELS = ["Conflict", "Duplicate", "Neutral"]


def _normalize_label(value: object) -> str:
    if value is None:
        return "Invalid"
    text = str(value).strip()
    if not text:
        return "Invalid"

    lowered = text.lower()
    if lowered in {"conflict", "contradiction"}:
        return "Conflict"
    if lowered in {"duplicate", "dup", "same"}:
        return "Duplicate"
    if lowered in {"neutral", "none", "unrelated", "compatible"}:
        return "Neutral"

    first = lowered.split()[0]
    if first == "conflict":
        return "Conflict"
    if first == "duplicate":
        return "Duplicate"
    if first == "neutral":
        return "Neutral"

    matches = re.findall(r"(?<![a-z])(conflict|duplicate|neutral)(?![a-z])", lowered)
    if matches:
        last = matches[-1]
        if last == "conflict":
            return "Conflict"
        if last == "duplicate":
            return "Duplicate"
        if last == "neutral":
            return "Neutral"

    last_idx: tuple[int, str] | None = None
    for lab in ("conflict", "duplicate", "neutral"):
        idx = lowered.rfind(lab)
        if idx >= 0 and (last_idx is None or idx > last_idx[0]):
            last_idx = (idx, lab)
    if last_idx is not None:
        if last_idx[1] == "conflict":
            return "Conflict"
        if last_idx[1] == "duplicate":
            return "Duplicate"
        if last_idx[1] == "neutral":
            return "Neutral"

    return "Invalid"


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                yield obj


def _expand_paths(items: list[str]) -> list[Path]:
    out: list[Path] = []
    for item in items:
        if any(ch in item for ch in "*?["):
            out.extend(Path(p) for p in glob.glob(item))
        else:
            out.append(Path(item))

    # de-dup while preserving order
    seen: set[Path] = set()
    unique: list[Path] = []
    for p in out:
        rp = p.resolve()
        if rp in seen:
            continue
        seen.add(rp)
        unique.append(p)
    return unique


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Join JSONL predictions back to the source CSV by row_index and output misclassifications. "
            "This is useful because demo_code.py does not store Text1/Text2 in JSONL."
        )
    )
    parser.add_argument(
        "--pred",
        action="append",
        required=True,
        help=(
            "Path to a JSONL prediction file. Can be passed multiple times. "
            "Globs are supported (e.g. 'predictions/*.jsonl')."
        ),
    )
    parser.add_argument(
        "--csv",
        default=None,
        help=(
            "Optional CSV path to use for all records. If omitted, uses each record's source_csv field."
        ),
    )
    parser.add_argument(
        "--out",
        default="errors.csv",
        help="Output CSV path (default: errors.csv)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Include correct predictions too (default: only mistakes)",
    )
    args = parser.parse_args()

    pred_paths = _expand_paths(args.pred)
    missing = [str(p) for p in pred_paths if not p.exists()]
    if missing:
        raise SystemExit("Missing files: " + ", ".join(missing))

    records: list[dict] = []
    for path in pred_paths:
        for rec in _iter_jsonl(path):
            if (
                "row_index" not in rec
                or "true_label" not in rec
                or "pred_label" not in rec
            ):
                continue
            try:
                row_index_raw = rec.get("row_index")
                if row_index_raw is None:
                    continue
                if isinstance(row_index_raw, int) and not isinstance(
                    row_index_raw, bool
                ):
                    row_index = row_index_raw
                elif isinstance(row_index_raw, str):
                    row_index = int(row_index_raw.strip())
                else:
                    row_index = int(str(row_index_raw))
            except Exception:
                continue

            source_csv = args.csv or rec.get("source_csv")
            if not source_csv:
                continue

            true_label = _normalize_label(rec.get("true_label"))
            pred_label = _normalize_label(rec.get("pred_label"))

            records.append(
                {
                    "source_csv": str(source_csv),
                    "row_index": row_index,
                    "true_label": true_label,
                    "pred_label": pred_label,
                    "correct": true_label == pred_label,
                    "backend": rec.get("backend"),
                    "model": rec.get("model"),
                    "prompt_name": rec.get("prompt_name"),
                    "prompt_id": rec.get("prompt_id"),
                }
            )

    if not records:
        raise SystemExit(
            "No usable records found (need row_index + true_label + pred_label)."
        )

    pred_df = pd.DataFrame.from_records(records)

    # Load each source CSV once
    csv_cache: dict[str, pd.DataFrame] = {}

    text1_list: list[str] = []
    text2_list: list[str] = []

    for _, row in pred_df.iterrows():
        source_csv = str(row["source_csv"])
        if source_csv not in csv_cache:
            df = pd.read_csv(
                source_csv, usecols=["Text1", "Text2", "Class"]
            ).reset_index(drop=True)
            csv_cache[source_csv] = df

        df = csv_cache[source_csv]
        idx = int(row["row_index"])
        if idx < 0 or idx >= len(df):
            text1_list.append("")
            text2_list.append("")
        else:
            text1_list.append(
                str(df.iloc[idx]["Text1"]) if df.iloc[idx]["Text1"] is not None else ""
            )
            text2_list.append(
                str(df.iloc[idx]["Text2"]) if df.iloc[idx]["Text2"] is not None else ""
            )

    pred_df.insert(2, "text1", text1_list)
    pred_df.insert(3, "text2", text2_list)

    if not args.all:
        pred_df = pred_df[pred_df["correct"] == False]  # noqa: E712

    pred_df = pred_df.sort_values(["source_csv", "row_index"]).reset_index(drop=True)

    out_path = Path(args.out)
    pred_df.to_csv(out_path, index=False)

    total = len(records)
    wrong = int(
        (pd.DataFrame.from_records(records)["correct"] == False).sum()
    )  # noqa: E712
    print(f"records: {total}")
    print(f"mistakes: {wrong}")
    print(f"wrote: {out_path}")


if __name__ == "__main__":
    main()
