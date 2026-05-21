from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Iterable

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

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

    # Some models may output extra tokens; try a simple first-token fallback.
    first = lowered.split()[0]
    if first in {"conflict"}:
        return "Conflict"
    if first in {"duplicate"}:
        return "Duplicate"
    if first in {"neutral"}:
        return "Neutral"

    return "Invalid"


def _iter_jsonl_records(path: Path) -> Iterable[dict]:
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


def _expand_paths(pred: list[str]) -> list[Path]:
    paths: list[Path] = []
    for p in pred:
        if any(ch in p for ch in "*?["):
            paths.extend(Path(x) for x in glob.glob(p))
        else:
            paths.append(Path(p))
    # de-dup while preserving order
    seen: set[Path] = set()
    out: list[Path] = []
    for p in paths:
        rp = p.resolve()
        if rp in seen:
            continue
        seen.add(rp)
        out.append(p)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute accuracy/precision/recall/F1 for JSONL predictions"
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
        "--average",
        choices=["macro", "micro", "weighted"],
        default="macro",
        help="Averaging for precision/recall/F1 (default: macro)",
    )
    parser.add_argument(
        "--include-invalid",
        action="store_true",
        help="Include invalid predicted labels as a separate class in reports",
    )
    args = parser.parse_args()

    paths = _expand_paths(args.pred)
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise SystemExit("Missing files: " + ", ".join(missing))

    y_true: list[str] = []
    y_pred: list[str] = []

    for path in paths:
        for rec in _iter_jsonl_records(path):
            if "true_label" not in rec or "pred_label" not in rec:
                continue
            y_true.append(_normalize_label(rec.get("true_label")))
            y_pred.append(_normalize_label(rec.get("pred_label")))

    if not y_true:
        raise SystemExit("No usable records found (need true_label + pred_label).")

    if args.include_invalid:
        labels = VALID_LABELS + ["Invalid"]
    else:
        labels = VALID_LABELS

        # If invalid predictions exist and we exclude them, count them as wrong by mapping
        # them to a non-existent label and filtering out from report labels.
        y_pred = [p if p in VALID_LABELS else "Invalid" for p in y_pred]

    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        average=args.average,
        zero_division=0,
    )

    print(f"n={len(y_true)}")
    print(f"accuracy:  {acc:.4f}")
    print(f"precision: {precision:.4f} ({args.average})")
    print(f"recall:    {recall:.4f} ({args.average})")
    print(f"f1-score:  {f1:.4f} ({args.average})")

    print("\nClassification report:")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=labels,
            zero_division=0,
        )
    )

    print("Confusion matrix (rows=true, cols=pred):")
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    print("labels:", labels)
    for row in cm.tolist():
        print(row)


if __name__ == "__main__":
    main()
