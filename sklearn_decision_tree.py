from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier


def _make_text(df: pd.DataFrame) -> pd.Series:
    t1 = df["Text1"].fillna("").astype(str)
    t2 = df["Text2"].fillna("").astype(str)
    return t1 + "\n" + t2


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a simple scikit-learn Decision Tree baseline and write JSONL predictions."
    )
    parser.add_argument(
        "--train", default="train.csv", help="Training CSV (default: train.csv)"
    )
    parser.add_argument(
        "--eval",
        action="append",
        required=True,
        help=(
            "Evaluation CSV path. Can be passed multiple times (e.g. --eval val.csv --eval test.csv)."
        ),
    )
    parser.add_argument(
        "--out-dir",
        default="predictions",
        help="Output directory for JSONL predictions (default: predictions)",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=20,
        help="Decision Tree max_depth (default: 20)",
    )
    parser.add_argument(
        "--min-samples-split",
        type=int,
        default=5,
        help="Decision Tree min_samples_split (default: 5)",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional limit for number of rows from each eval CSV (debugging only)",
    )
    args = parser.parse_args()

    train_path = Path(args.train)
    if not train_path.exists():
        raise SystemExit(f"Missing train CSV: {train_path}")

    train_df = pd.read_csv(train_path, usecols=["Text1", "Text2", "Class"]).reset_index(
        drop=True
    )
    x_train = _make_text(train_df)
    y_train = train_df["Class"].fillna("").astype(str)

    model = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer()),
            (
                "clf",
                DecisionTreeClassifier(
                    random_state=42,
                    max_depth=args.max_depth,
                    min_samples_split=args.min_samples_split,
                ),
            ),
        ]
    )
    model.fit(x_train, y_train)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for eval_csv in args.eval:
        eval_path = Path(eval_csv)
        if not eval_path.exists():
            raise SystemExit(f"Missing eval CSV: {eval_path}")

        eval_df = pd.read_csv(
            eval_path, usecols=["Text1", "Text2", "Class"]
        ).reset_index(drop=True)
        if args.max_rows is not None:
            eval_df = eval_df.head(args.max_rows).reset_index(drop=True)

        x_eval = _make_text(eval_df)
        y_true = eval_df["Class"].fillna("").astype(str).tolist()
        y_pred = model.predict(x_eval).tolist()

        out_path = out_dir / f"{eval_path.name}__sklearn__decision_tree.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for row_index, (true_label, pred_label) in enumerate(zip(y_true, y_pred)):
                rec = {
                    "source_csv": str(eval_path),
                    "row_index": row_index,
                    "true_label": true_label,
                    "pred_label": pred_label,
                    "backend": "sklearn",
                    "model": "decision_tree",
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        print(f"wrote: {out_path} (n={len(eval_df)})")


if __name__ == "__main__":
    main()
