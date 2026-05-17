from __future__ import annotations

import argparse
from collections import Counter
import random
from typing import Literal

import pandas as pd
from sklearn.model_selection import train_test_split


def _print_distribution(name: str, labels: pd.Series) -> None:
    counts = labels.value_counts(dropna=False)
    total = int(counts.sum())

    print(f"\n{name} split (n={total})")
    for cls, count in counts.items():
        pct = (float(count) / total) * 100 if total else 0.0
        print(f"- {cls}: {int(count)} ({pct:.2f}%)")


def _row_stratified_split(
    df: pd.DataFrame,
    *,
    train_size: float,
    test_size: float,
    val_size: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # Two-stage stratified split to achieve train/test/val.
    temp_size = test_size + val_size
    train_df, temp_df = train_test_split(
        df,
        test_size=temp_size,
        random_state=seed,
        stratify=df["Class"],
    )

    # Split temp into test/val keeping their relative proportions.
    test_fraction_of_temp = test_size / temp_size
    test_df, val_df = train_test_split(
        temp_df,
        test_size=(1.0 - test_fraction_of_temp),
        random_state=seed,
        stratify=temp_df["Class"],
    )
    return (
        train_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
    )


def _group_split_by_text1(
    df: pd.DataFrame,
    *,
    train_size: float,
    test_size: float,
    val_size: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # Split by GroupId (Text1 blocks) to avoid leakage.
    splits: list[Literal["train", "test", "val"]] = ["train", "test", "val"]

    classes = sorted(df["Class"].unique().tolist())
    overall_counts = (
        df["Class"].value_counts().reindex(classes, fill_value=0).astype(float)
    )

    target_rows = {
        "train": float(len(df)) * train_size,
        "test": float(len(df)) * test_size,
        "val": float(len(df)) * val_size,
    }
    target_counts = {
        "train": overall_counts * train_size,
        "test": overall_counts * test_size,
        "val": overall_counts * val_size,
    }

    group_class = pd.crosstab(df["GroupId"], df["Class"]).reindex(
        columns=classes, fill_value=0
    )
    group_sizes = group_class.sum(axis=1).astype(int)

    group_ids = group_class.index.tolist()
    rng = random.Random(seed)
    rng.shuffle(group_ids)
    group_ids.sort(key=lambda gid: int(group_sizes.loc[gid]), reverse=True)

    assigned: dict[str, list[int]] = {"train": [], "test": [], "val": []}
    current_rows = {"train": 0, "test": 0, "val": 0}
    current_counts = {
        "train": pd.Series(0.0, index=classes),
        "test": pd.Series(0.0, index=classes),
        "val": pd.Series(0.0, index=classes),
    }

    # Greedy assignment: place largest groups first to match target size + class distribution.
    for gid in group_ids:
        g_counts = group_class.loc[gid].astype(float)
        g_rows = int(group_sizes.loc[gid])

        best_split: str | None = None
        best_cost: float | None = None

        for candidate in splits:
            # Evaluate TOTAL objective across all splits, not just the candidate split.
            # Otherwise small splits look artificially attractive at the beginning.
            total_row_cost = 0.0
            total_class_cost = 0.0
            total_overflow_cost = 0.0

            for split in splits:
                add_rows = g_rows if split == candidate else 0
                add_counts = g_counts if split == candidate else 0

                new_rows = current_rows[split] + add_rows
                new_counts = current_counts[split] + add_counts

                total_row_cost += abs(new_rows - target_rows[split]) / max(
                    1.0, target_rows[split]
                )
                total_class_cost += float(
                    (new_counts - target_counts[split]).abs().sum()
                ) / max(1.0, float(len(df)))

                overflow = max(0.0, new_rows - target_rows[split])
                total_overflow_cost += overflow / max(1.0, float(len(df)))

            # Weights: row size is primary; class distribution is secondary; overflow is a mild extra.
            cost = (
                (1.0 * total_row_cost)
                + (0.35 * total_class_cost)
                + (0.5 * total_overflow_cost)
            )

            if best_cost is None or cost < best_cost:
                best_cost = cost
                best_split = candidate
            elif cost == best_cost:
                # Deterministic tie-break using seeded RNG.
                if rng.random() < 0.5:
                    best_split = candidate

        assert best_split is not None
        assigned[best_split].append(int(gid))
        current_rows[best_split] += g_rows
        current_counts[best_split] = current_counts[best_split] + g_counts

    train_df = df[df["GroupId"].isin(assigned["train"])].reset_index(drop=True)
    test_df = df[df["GroupId"].isin(assigned["test"])].reset_index(drop=True)
    val_df = df[df["GroupId"].isin(assigned["val"])].reset_index(drop=True)
    return train_df, test_df, val_df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="70/20/10 train/test/val split for cdn_pairs.csv (optionally grouped by Text1)"
    )
    parser.add_argument("--csv", default="cdn_pairs.csv", help="Path to input CSV")
    parser.add_argument(
        "--train-size",
        type=float,
        default=0.7,
        help="Train split fraction (default: 0.7)",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test split fraction (default: 0.2)",
    )
    parser.add_argument(
        "--val-size",
        type=float,
        default=0.1,
        help="Validation split fraction (default: 0.1)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--group-by",
        choices=["text1", "none"],
        default="text1",
        help="How to split: keep same Text1 in one split (text1) or regular row-level stratified (none). Default: text1",
    )
    parser.add_argument(
        "--train-out", default="train.csv", help="Path to save train split CSV"
    )
    parser.add_argument(
        "--test-out", default="test.csv", help="Path to save test split CSV"
    )
    parser.add_argument(
        "--val-out", default="val.csv", help="Path to save validation split CSV"
    )
    args = parser.parse_args()

    if args.train_size <= 0 or args.test_size <= 0 or args.val_size <= 0:
        raise ValueError("train/test/val sizes must be > 0")
    total = float(args.train_size) + float(args.test_size) + float(args.val_size)
    if abs(total - 1.0) > 1e-9:
        raise ValueError(
            f"train/test/val sizes must sum to 1.0, got {args.train_size}+{args.test_size}+{args.val_size}={total}"
        )

    df = pd.read_csv(args.csv)
    required = {"Text1", "Text2", "Class"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in CSV: {sorted(missing)}")

    # Keep only needed columns for now (avoids surprises if extra columns exist)
    df = (
        df[["Text1", "Text2", "Class"]]
        .dropna(subset=["Text1", "Text2", "Class"])
        .reset_index(drop=True)
    )

    # Grouping helpers: the dataset is often organized in Text1 blocks.
    # GroupId: 1..N for each distinct Text1.
    df["GroupId"] = pd.factorize(df["Text1"])[0] + 1
    df["RowInGroup"] = df.groupby("GroupId").cumcount() + 1

    # Sanity check: stratify requires at least 2 samples per class (in practice).
    class_counts = Counter(df["Class"].tolist())
    too_small = [cls for cls, n in class_counts.items() if n < 2]
    if too_small:
        raise ValueError(
            "Some classes have <2 samples; stratified split is not possible: "
            + ", ".join(sorted(too_small))
        )

    if args.group_by == "none":
        train_df, test_df, val_df = _row_stratified_split(
            df,
            train_size=args.train_size,
            test_size=args.test_size,
            val_size=args.val_size,
            seed=args.seed,
        )
    else:
        train_df, test_df, val_df = _group_split_by_text1(
            df,
            train_size=args.train_size,
            test_size=args.test_size,
            val_size=args.val_size,
            seed=args.seed,
        )

    train_df.to_csv(args.train_out, index=False)
    test_df.to_csv(args.test_out, index=False)
    val_df.to_csv(args.val_out, index=False)
    print(f"\nSaved TRAIN to {args.train_out}")
    print(f"Saved TEST  to {args.test_out}")
    print(f"Saved VAL   to {args.val_out}")

    _print_distribution("TRAIN", train_df["Class"])
    _print_distribution("TEST", test_df["Class"])
    _print_distribution("VAL", val_df["Class"])


if __name__ == "__main__":
    main()
