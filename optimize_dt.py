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


def _decode_solution(
    position: np.ndarray,
    bounds: Bounds,
) -> tuple[int, int, int, float, int, int | None]:
    """Map continuous PSO position to Decision Tree hyperparameters.

    Integer parameters are rounded to nearest int. max_leaf_nodes uses sentinel:
    values <= 0 are treated as None (unlimited).
    """
    clipped = _clip(position, bounds)

    max_depth = int(np.rint(clipped[0]))
    min_samples_split = int(np.rint(clipped[1]))
    min_samples_leaf = int(np.rint(clipped[2]))
    min_weight_fraction_leaf = float(clipped[3])
    random_state = int(np.rint(clipped[4]))
    max_leaf_nodes_raw = int(np.rint(clipped[5]))

    # Safety clamp (should already be within bounds).
    max_depth = max(int(bounds.low[0]), min(max_depth, int(bounds.high[0])))
    min_samples_split = max(
        int(bounds.low[1]), min(min_samples_split, int(bounds.high[1]))
    )

    min_samples_leaf = max(
        int(bounds.low[2]), min(min_samples_leaf, int(bounds.high[2]))
    )
    min_weight_fraction_leaf = max(
        float(bounds.low[3]), min(min_weight_fraction_leaf, float(bounds.high[3]))
    )
    random_state = max(int(bounds.low[4]), min(random_state, int(bounds.high[4])))
    max_leaf_nodes_raw = max(
        int(bounds.low[5]), min(max_leaf_nodes_raw, int(bounds.high[5]))
    )

    max_leaf_nodes = None if max_leaf_nodes_raw <= 0 else max_leaf_nodes_raw

    return (
        max_depth,
        min_samples_split,
        min_samples_leaf,
        min_weight_fraction_leaf,
        random_state,
        max_leaf_nodes,
    )


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
        "--max-depth-max", type=int, default=200, help="Max max_depth (default: 200)"
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

    parser.add_argument(
        "--min-leaf-min",
        type=int,
        default=1,
        help="Min min_samples_leaf (default: 1)",
    )
    parser.add_argument(
        "--min-leaf-max",
        type=int,
        default=30,
        help="Max min_samples_leaf (default: 30)",
    )
    parser.add_argument(
        "--min-weight-frac-min",
        type=float,
        default=0.0,
        help="Min min_weight_fraction_leaf (default: 0.0)",
    )
    parser.add_argument(
        "--min-weight-frac-max",
        type=float,
        default=0.1,
        help="Max min_weight_fraction_leaf (default: 0.1)",
    )
    parser.add_argument(
        "--random-state-min",
        type=int,
        default=0,
        help="Min random_state (default: 0)",
    )
    parser.add_argument(
        "--random-state-max",
        type=int,
        default=9999,
        help="Max random_state (default: 9999)",
    )
    parser.add_argument(
        "--max-leaf-nodes-min",
        type=int,
        default=0,
        help="Min max_leaf_nodes (default: 0; 0 means None/unlimited)",
    )
    parser.add_argument(
        "--max-leaf-nodes-max",
        type=int,
        default=500,
        help="Max max_leaf_nodes (default: 500)",
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

    print(
        "Search space: "
        f"max_depth=[{args.max_depth_min}, {args.max_depth_max}], "
        f"min_samples_split=[{args.min_split_min}, {args.min_split_max}], "
        f"min_samples_leaf=[{args.min_leaf_min}, {args.min_leaf_max}], "
        f"min_weight_fraction_leaf=[{args.min_weight_frac_min}, {args.min_weight_frac_max}], "
        f"random_state=[{args.random_state_min}, {args.random_state_max}], "
        f"max_leaf_nodes=[{args.max_leaf_nodes_min}, {args.max_leaf_nodes_max}] (0=>None)"
    )

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
    if args.min_leaf_min > args.min_leaf_max:
        raise SystemExit("Invalid bounds: min-leaf-min > min-leaf-max")
    if args.min_weight_frac_min > args.min_weight_frac_max:
        raise SystemExit("Invalid bounds: min-weight-frac-min > min-weight-frac-max")
    if args.random_state_min > args.random_state_max:
        raise SystemExit("Invalid bounds: random-state-min > random-state-max")
    if args.max_leaf_nodes_min > args.max_leaf_nodes_max:
        raise SystemExit("Invalid bounds: max-leaf-nodes-min > max-leaf-nodes-max")

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
        low=np.array(
            [
                args.max_depth_min,
                args.min_split_min,
                args.min_leaf_min,
                args.min_weight_frac_min,
                args.random_state_min,
                args.max_leaf_nodes_min,
            ],
            dtype=float,
        ),
        high=np.array(
            [
                args.max_depth_max,
                args.min_split_max,
                args.min_leaf_max,
                args.min_weight_frac_max,
                args.random_state_max,
                args.max_leaf_nodes_max,
            ],
            dtype=float,
        ),
    )

    def fitness(position: np.ndarray) -> float:
        """PSO minimizes: return 1 - macroF1."""
        (
            max_depth,
            min_samples_split,
            min_samples_leaf,
            min_weight_fraction_leaf,
            random_state,
            max_leaf_nodes,
        ) = _decode_solution(position, bounds)

        clf = DecisionTreeClassifier(
            random_state=random_state,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_leaf_nodes=max_leaf_nodes,
        )
        clf.fit(x_train, y_train)
        pred = clf.predict(x_val)

        score = f1_score(y_val, pred, average="macro", zero_division=0)
        return 1.0 - float(score)

    # --- PSO (global best) ---
    n_particles = int(args.particles)
    n_iters = int(args.iters)
    dim = int(bounds.low.shape[0])

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

        (
            best_max_depth,
            best_min_split,
            best_min_leaf,
            best_min_weight_frac,
            best_random_state,
            best_max_leaf_nodes,
        ) = _decode_solution(gbest_pos, bounds)
        best_f1 = 1.0 - gbest_cost
        print(
            f"iter {it + 1:02d}/{n_iters}: best macro-F1={best_f1:.4f} "
            f"(max_depth={best_max_depth}, min_samples_split={best_min_split}, "
            f"min_samples_leaf={best_min_leaf}, min_weight_fraction_leaf={best_min_weight_frac:.4f}, "
            f"random_state={best_random_state}, max_leaf_nodes={best_max_leaf_nodes})"
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

    (
        best_max_depth,
        best_min_split,
        best_min_leaf,
        best_min_weight_frac,
        best_random_state,
        best_max_leaf_nodes,
    ) = _decode_solution(gbest_pos, bounds)
    best_macro_f1 = 1.0 - gbest_cost

    print("\n=== Best solution ===")
    print(f"max_depth: {best_max_depth}")
    print(f"min_samples_split: {best_min_split}")
    print(f"min_samples_leaf: {best_min_leaf}")
    print(f"min_weight_fraction_leaf: {best_min_weight_frac:.6f}")
    print(f"random_state: {best_random_state}")
    print(f"max_leaf_nodes: {best_max_leaf_nodes}")
    print(f"best macro-F1: {best_macro_f1:.4f}")


if __name__ == "__main__":
    main()
