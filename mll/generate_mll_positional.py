#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

from .generate import generate_dataset


def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--n", type=int, default=500000)
    ap.add_argument("--out", type=Path, default=Path("mll_positional.txt"))
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--min-leaves", type=int, default=2)
    ap.add_argument("--max-leaves", type=int, default=20)
    ap.add_argument(
        "--n-atoms",
        type=int,
        default=10,
        help="atom name pool size; each atomic axiom draws uniformly in 0..n_atoms-1",
    )

    ap.add_argument(
        "--one-prob",
        type=float,
        default=0.15,
        help="probability of a 𝟙 leaf; residual after one_prob+ax_prob is ⊢ 𝟙, ⊥",
    )
    ap.add_argument(
        "--ax-prob",
        type=float,
        default=0.80,
        help="probability of an atomic axiom leaf; residual after one_prob+ax_prob is ⊢ 𝟙, ⊥",
    )
    ap.add_argument(
        "--ax-pos-prob",
        type=float,
        default=0.50,
        help="given an atomic axiom, probability of ⊢ A, A⊥ (else ⊢ A⊥, A)",
    )
    ap.add_argument(
        "--tensor-prob",
        type=float,
        default=0.45,
        help=(
            "probability of ⊗ when the forest has size > 1; "
            "residual of tensor_prob+par_prob is bot_prob for ⊥; "
            "if ⅋ is unavailable, par_prob mass is transferred to ⊗"
        ),
    )
    ap.add_argument(
        "--par-prob",
        type=float,
        default=0.45,
        help=(
            "probability of ⅋ when available; "
            "residual of tensor_prob+par_prob is bot_prob for ⊥; "
            "if ⅋ is unavailable with several trees, this mass goes to ⊗"
        ),
    )
    ap.add_argument(
        "--stop-prob",
        type=float,
        default=0.85,
        help=(
            "when the forest has a single tree and stopping is allowed, "
            "probability of returning immediately; otherwise continue with ⅋/⊥"
        ),
    )

    ap.add_argument("--single-formula", action="store_true")

    args = ap.parse_args()

    generate_dataset(
        out=args.out,
        n=args.n,
        seed=args.seed,
        min_leaves=args.min_leaves,
        max_leaves=args.max_leaves,
        n_atoms=args.n_atoms,
        one_prob=args.one_prob,
        ax_prob=args.ax_prob,
        ax_pos_prob=args.ax_pos_prob,
        tensor_prob=args.tensor_prob,
        par_prob=args.par_prob,
        stop_prob=args.stop_prob,
        single_formula=args.single_formula,
    )

    print(f"{args.n} proofs written to {args.out}")


if __name__ == "__main__":
    main()
