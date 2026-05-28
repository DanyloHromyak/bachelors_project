from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.tree import DecisionTreeClassifier


def _make_text(df: pd.DataFrame) -> pd.Series:
    t1 = df["Text1"].fillna("").astype(str)
    t2 = df["Text2"].fillna("").astype(str)
    return t1 + "\n" + t2


@dataclass(frozen=True)
class Bounds:
    low: np.ndarray
    high: np.ndarray


def _clip(x: np.ndarray, bounds: Bounds) -> np.ndarray:
    return np.minimum(np.maximum(x, bounds.low), bounds.high)


def _decode_solution(position: np.ndarray, bounds: Bounds) -> tuple[int, int]:
    """Map continuous PSO position to discrete Decision Tree hyperparameters."""
    clipped = _clip(position, bounds)

    max_depth = int(np.rint(clipped[0]))
    min_samples_split = int(np.rint(clipped[1]))

    # Safety clamp (should already be within bounds).
    max_depth = max(int(bounds.low[0]), min(max_depth, int(bounds.high[0])))
    min_samples_split = max(
        int(bounds.low[1]), min(min_samples_split, int(bounds.high[1]))
    )

    return max_depth, min_samples_split


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Hyperparameter optimization for DecisionTreeClassifier using Particle Swarm Optimization (PSO)."
        )
    )
    parser.add_argument(
        "--train", default="train.csv", help="Training CSV path (default: train.csv)"
    )
    parser.add_argument(
        "--val", default="val.csv", help="Validation CSV path (default: val.csv)"
    )

    # Hyperparameter bounds (continuous PSO values will be rounded to ints)
    parser.add_argument(
        "--max-depth-min", type=int, default=3, help="Min max_depth (default: 3)"
    )
    parser.add_argument(
        "--max-depth-max", type=int, default=20, help="Max max_depth (default: 20)"
    )
    parser.add_argument(
        "--min-split-min",
        type=int,
        default=2,
        help="Min min_samples_split (default: 2)",
    )
    parser.add_argument(
        "--min-split-max",
        type=int,
        default=15,
        help="Max min_samples_split (default: 15)",
    )

    # PSO settings
    parser.add_argument(
        "--particles", type=int, default=20, help="#solutions/particles (default: 20)"
    )
    parser.add_argument(
        "--iters", type=int, default=20, help="#iterations (default: 20)"
    )
    parser.add_argument(
        "--w", type=float, default=0.7, help="Inertia weight w (default: 0.7)"
    )
    parser.add_argument(
        "--c1", type=float, default=1.4, help="Cognitive coefficient c1 (default: 1.4)"
    )
    parser.add_argument(
        "--c2", type=float, default=1.4, help="Social coefficient c2 (default: 1.4)"
    )
    parser.add_argument("--seed", type=int, default=42, help="RNG seed (default: 42)")

    args = parser.parse_args()

    train_path = Path(args.train)
    val_path = Path(args.val)
    if not train_path.exists():
        raise SystemExit(f"Missing train CSV: {train_path}")
    if not val_path.exists():
        raise SystemExit(f"Missing val CSV: {val_path}")

    if args.max_depth_min > args.max_depth_max:
        raise SystemExit("Invalid bounds: max-depth-min > max-depth-max")
    if args.min_split_min > args.min_split_max:
        raise SystemExit("Invalid bounds: min-split-min > min-split-max")

    rng = np.random.default_rng(args.seed)

    # Load data
    train_df = pd.read_csv(train_path, usecols=["Text1", "Text2", "Class"]).reset_index(
        drop=True
    )
    val_df = pd.read_csv(val_path, usecols=["Text1", "Text2", "Class"]).reset_index(
        drop=True
    )

    x_train_text = _make_text(train_df)
    y_train = train_df["Class"].fillna("").astype(str).to_numpy()

    x_val_text = _make_text(val_df)
    y_val = val_df["Class"].fillna("").astype(str).to_numpy()

    # Vectorize (fit on train only)
    vectorizer = TfidfVectorizer()
    x_train = vectorizer.fit_transform(x_train_text)
    x_val = vectorizer.transform(x_val_text)

    bounds = Bounds(
        low=np.array([args.max_depth_min, args.min_split_min], dtype=float),
        high=np.array([args.max_depth_max, args.min_split_max], dtype=float),
    )

    def fitness(position: np.ndarray) -> float:
        """PSO minimizes: return 1 - macroF1."""
        max_depth, min_samples_split = _decode_solution(position, bounds)

        clf = DecisionTreeClassifier(
            random_state=args.seed,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
        )
        clf.fit(x_train, y_train)
        pred = clf.predict(x_val)

        score = f1_score(y_val, pred, average="macro", zero_division=0)
        return 1.0 - float(score)

    # --- PSO (global best) ---
    n_particles = int(args.particles)
    n_iters = int(args.iters)
    dim = 2

    # Initialize swarm positions uniformly in bounds
    pos = rng.uniform(bounds.low, bounds.high, size=(n_particles, dim))

    # Initialize velocities in a small range based on bounds span
    span = bounds.high - bounds.low
    vel = rng.uniform(-0.1 * span, 0.1 * span, size=(n_particles, dim))

    # Personal and global bests
    pbest_pos = pos.copy()
    pbest_cost = np.full((n_particles,), np.inf, dtype=float)

    gbest_pos = pos[0].copy()
    gbest_cost = np.inf

    w = float(args.w)
    c1 = float(args.c1)
    c2 = float(args.c2)

    for it in range(n_iters):
        # Evaluate
        costs = np.array([fitness(pos[i]) for i in range(n_particles)], dtype=float)

        # Update personal best
        improved = costs < pbest_cost
        pbest_cost[improved] = costs[improved]
        pbest_pos[improved] = pos[improved]

        # Update global best
        best_idx = int(np.argmin(pbest_cost))
        if pbest_cost[best_idx] < gbest_cost:
            gbest_cost = float(pbest_cost[best_idx])
            gbest_pos = pbest_pos[best_idx].copy()

        best_max_depth, best_min_split = _decode_solution(gbest_pos, bounds)
        best_f1 = 1.0 - gbest_cost
        print(
            f"iter {it + 1:02d}/{n_iters}: best macro-F1={best_f1:.4f} "
            f"(max_depth={best_max_depth}, min_samples_split={best_min_split})"
        )

        # Velocity and position update
        r1 = rng.random(size=(n_particles, dim))
        r2 = rng.random(size=(n_particles, dim))

        cognitive = c1 * r1 * (pbest_pos - pos)
        social = c2 * r2 * (gbest_pos - pos)
        vel = w * vel + cognitive + social
        pos = pos + vel

        # Keep positions within bounds
        pos = _clip(pos, bounds)

    best_max_depth, best_min_split = _decode_solution(gbest_pos, bounds)
    best_macro_f1 = 1.0 - gbest_cost

    print("\n=== Best solution ===")
    print(f"max_depth: {best_max_depth}")
    print(f"min_samples_split: {best_min_split}")
    print(f"best macro-F1: {best_macro_f1:.4f}")


if __name__ == "__main__":
    main()
