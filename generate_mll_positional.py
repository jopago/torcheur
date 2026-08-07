#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

# =========================
# Formulas
# =========================


@dataclass(frozen=True)
class Atom:
    name: str
    neg: bool = False


@dataclass(frozen=True)
class One:
    pass


@dataclass(frozen=True)
class Bottom:
    pass


@dataclass(frozen=True)
class Tensor:
    left: Formula
    right: Formula


@dataclass(frozen=True)
class Par:
    left: Formula
    right: Formula


Formula = Atom | One | Bottom | Tensor | Par
Sequent = tuple[Formula, ...]


def atom_name(index: int) -> str:
    """Letter-only names: a, b, ..., z, aa, ab, ... (no digits)."""
    name = ""
    n = index
    while True:
        name = chr(ord("a") + n % 26) + name
        n = n // 26 - 1
        if n < 0:
            return name


def dual(f: Formula) -> Formula:
    if isinstance(f, Atom):
        return Atom(f.name, not f.neg)
    if isinstance(f, One):
        return Bottom()
    if isinstance(f, Bottom):
        return One()
    if isinstance(f, Tensor):
        return Par(dual(f.left), dual(f.right))
    if isinstance(f, Par):
        return Tensor(dual(f.left), dual(f.right))
    raise TypeError(type(f))


def fstr(f: Formula) -> str:
    if isinstance(f, Atom):
        return f"{f.name}⊥" if f.neg else f.name
    if isinstance(f, One):
        return "𝟙"
    if isinstance(f, Bottom):
        return "⊥"
    if isinstance(f, Tensor):
        return f"({fstr(f.left)} ⊗ {fstr(f.right)})"
    if isinstance(f, Par):
        return f"({fstr(f.left)} ⅋ {fstr(f.right)})"
    raise TypeError(type(f))


def sstr(seq: Sequent) -> str:
    return "⊢ " + ", ".join(fstr(x) for x in seq)


# =========================
# Internal proofs
# =========================


@dataclass(frozen=True)
class Ax:
    sequent: Sequent


@dataclass(frozen=True)
class OneRule:
    sequent: Sequent


@dataclass(frozen=True)
class BotRule:
    sequent: Sequent
    index: int
    child: Proof


@dataclass(frozen=True)
class ParRule:
    sequent: Sequent
    index: int
    child: Proof


@dataclass(frozen=True)
class TensorRule:
    sequent: Sequent
    index: int
    left_positions: tuple[int, ...]
    left: Proof
    right: Proof


Proof = Ax | OneRule | BotRule | ParRule | TensorRule


# =========================
# Premise reconstruction
# (conclusion → premises)
# =========================


def tensor_premises(
    seq: Sequent,
    principal_index: int,
    left_positions: tuple[int, ...],
) -> tuple[Sequent, Sequent]:

    if not (0 <= principal_index < len(seq)):
        raise ValueError("invalid principal_index")

    principal = seq[principal_index]
    if not isinstance(principal, Tensor):
        raise TypeError("principal formula is not a tensor")

    if tuple(sorted(left_positions)) != left_positions:
        raise ValueError("left_positions must be sorted")

    # Sorted ⇒ endpoints are min / max; enough to check bounds.
    if left_positions and (left_positions[0] < 0 or left_positions[-1] >= len(seq)):
        raise IndexError("branch index out of sequent")

    left_set = set(left_positions)
    if len(left_set) != len(left_positions):
        raise ValueError("duplicate in left_positions")

    if principal_index in left_set:
        raise ValueError("principal index belongs to no branch")

    left_sequent: list[Formula] = []
    right_sequent: list[Formula] = []

    for j, formula in enumerate(seq):
        if j == principal_index:
            left_sequent.append(principal.left)
            right_sequent.append(principal.right)
        elif j in left_set:
            left_sequent.append(formula)
        else:
            right_sequent.append(formula)

    return tuple(left_sequent), tuple(right_sequent)


# =========================
# Proof constructors
# (premises → conclusion)
# =========================


def make_atom_ax(name: str) -> Ax:
    atom = Atom(name)
    return Ax((atom, dual(atom)))


def make_unit_ax() -> Ax:
    return Ax((One(), Bottom()))


def make_one() -> OneRule:
    return OneRule((One(),))


def make_bot(child: Proof, rng: random.Random) -> BotRule:
    index = rng.randrange(len(child.sequent) + 1)
    sequent = child.sequent[:index] + (Bottom(),) + child.sequent[index:]
    return BotRule(sequent, index, child)


def make_par(child: Proof, rng: random.Random) -> ParRule:
    if len(child.sequent) < 2:
        raise ValueError("par requires >= 2 formulas")

    index = rng.randrange(len(child.sequent) - 1)
    left_formula = child.sequent[index]
    right_formula = child.sequent[index + 1]
    sequent = (
        child.sequent[:index]
        + (Par(left_formula, right_formula),)
        + child.sequent[index + 2 :]
    )
    return ParRule(sequent, index, child)


def make_tensor(left: Proof, right: Proof, rng: random.Random) -> TensorRule:
    if not left.sequent or not right.sequent:
        raise ValueError("tensor requires two non-empty premises")

    left_factor_index = rng.randrange(len(left.sequent))
    right_factor_index = rng.randrange(len(right.sequent))

    left_factor = left.sequent[left_factor_index]
    right_factor = right.sequent[right_factor_index]

    gamma1 = left.sequent[:left_factor_index]
    gamma2 = left.sequent[left_factor_index + 1 :]
    delta1 = right.sequent[:right_factor_index]
    delta2 = right.sequent[right_factor_index + 1 :]

    sequent = gamma1 + delta1 + (Tensor(left_factor, right_factor),) + gamma2 + delta2
    principal_index = len(gamma1) + len(delta1)

    left_positions = tuple(range(len(gamma1))) + tuple(
        range(principal_index + 1, principal_index + 1 + len(gamma2))
    )

    return TensorRule(
        sequent=sequent,
        index=principal_index,
        left_positions=left_positions,
        left=left,
        right=right,
    )


# =========================
# Generation
# =========================


def par_candidates(forest: list[Proof]) -> list[int]:
    return [i for i, proof in enumerate(forest) if len(proof.sequent) >= 2]


def generate_proof(
    rng: random.Random,
    n_leaves: int,
    n_atoms: int,
    one_prob: float,
    ax_prob: float,
    tensor_prob: float,
    par_prob: float,
    single_formula: bool,
) -> Proof:
    if n_atoms < 1:
        raise ValueError("n_atoms must be >= 1")
    if one_prob < 0 or ax_prob < 0 or one_prob + ax_prob > 1:
        raise ValueError("need one_prob >= 0, ax_prob >= 0, one_prob + ax_prob <= 1")
    if tensor_prob < 0 or par_prob < 0 or tensor_prob + par_prob > 1:
        raise ValueError(
            "need tensor_prob >= 0, par_prob >= 0, tensor_prob + par_prob <= 1"
        )

    # Residual mass is bot: bot_prob = 1 - tensor_prob - par_prob.
    forest: list[Proof] = []

    for _ in range(n_leaves):
        roll = rng.random()
        if roll < one_prob:
            forest.append(make_one())
        elif roll < one_prob + ax_prob:
            forest.append(make_atom_ax(atom_name(rng.randrange(n_atoms))))
        else:
            forest.append(make_unit_ax())

    while True:
        can_par = bool(par_candidates(forest))
        # single_formula + sequent longer than 1 ⇒ that sequent has ≥2 formulas ⇒ can_par
        must_continue = (
            single_formula and len(forest) == 1 and len(forest[0].sequent) > 1
        )

        roll = rng.random()

        if len(forest) > 1:
            if roll < tensor_prob:
                left_index = rng.randrange(len(forest))
                left = forest.pop(left_index)
                right_index = rng.randrange(len(forest))
                right = forest.pop(right_index)
                forest.append(make_tensor(left, right, rng))
                continue

            if roll < tensor_prob + par_prob and can_par:
                index = rng.choice(par_candidates(forest))
                forest[index] = make_par(forest[index], rng)
                continue

            # bot residual, or par mass when par is unavailable
            index = rng.randrange(len(forest))
            forest[index] = make_bot(forest[index], rng)
            continue

        # Forest size 1: tensor_prob becomes stop probability.
        if must_continue:
            forest[0] = make_par(forest[0], rng)
            continue

        if roll < tensor_prob:
            return forest[0]

        if roll < tensor_prob + par_prob and can_par:
            forest[0] = make_par(forest[0], rng)
            continue

        forest[0] = make_bot(forest[0], rng)


# =========================
# Independent checker
# =========================


def check(proof: Proof) -> bool:

    if isinstance(proof, Ax):
        if len(proof.sequent) != 2:
            return False
        left_formula, right_formula = proof.sequent
        return (
            dual(left_formula) == right_formula
        )  # no need that the formulas are atomic

    if isinstance(proof, OneRule):
        return proof.sequent == (One(),)

    if isinstance(proof, BotRule):
        index = proof.index
        if not (0 <= index < len(proof.sequent)):
            return False
        if not isinstance(proof.sequent[index], Bottom):
            return False
        expected = proof.sequent[:index] + proof.sequent[index + 1 :]
        return expected == proof.child.sequent and check(proof.child)

    if isinstance(proof, ParRule):
        index = proof.index
        if not (0 <= index < len(proof.sequent)):
            return False
        principal = proof.sequent[index]
        if not isinstance(principal, Par):
            return False

        expected = (
            proof.sequent[:index]
            + (principal.left, principal.right)
            + proof.sequent[index + 1 :]
        )
        return expected == proof.child.sequent and check(proof.child)

    if isinstance(proof, TensorRule):
        if not (0 <= proof.index < len(proof.sequent)):
            return False

        try:
            expected_left, expected_right = tensor_premises(
                proof.sequent,
                proof.index,
                proof.left_positions,
            )
        except ValueError:
            return False

        return (
            expected_left == proof.left.sequent
            and expected_right == proof.right.sequent
            and check(proof.left)
            and check(proof.right)
        )

    return False


# =========================
# Compact format
# =========================


def positions_str(positions: tuple[int, ...]) -> str:
    return "[" + ",".join(map(str, positions)) + "]"


def proof_str(proof: Proof) -> str:
    if isinstance(proof, Ax):
        return "ax"
    if isinstance(proof, OneRule):
        return "𝟙"
    if isinstance(proof, BotRule):
        return f"⊥({proof.index},{proof_str(proof.child)})"
    if isinstance(proof, ParRule):
        return f"⅋({proof.index},{proof_str(proof.child)})"
    if isinstance(proof, TensorRule):
        return (
            f"⊗({proof.index},{positions_str(proof.left_positions)},"
            f"{proof_str(proof.left)},{proof_str(proof.right)})"
        )
    raise TypeError(type(proof))


def line_str(proof: Proof) -> str:
    if not check(proof):
        raise RuntimeError("generated invalid proof")
    sequent = json.dumps(sstr(proof.sequent), ensure_ascii=False)
    return f"{sequent} || {proof_str(proof)}"


# =========================
# Dataset
# =========================


def generate_dataset(
    out: Path,
    n: int,
    seed: int,
    min_leaves: int,
    max_leaves: int,
    n_atoms: int,
    one_prob: float,
    ax_prob: float,
    tensor_prob: float,
    par_prob: float,
    single_formula: bool,
) -> None:

    rng = random.Random(seed)

    with out.open("w", encoding="utf-8") as f:
        for _ in range(n):
            proof = generate_proof(
                rng=rng,
                n_leaves=rng.randint(min_leaves, max_leaves),
                n_atoms=n_atoms,
                one_prob=one_prob,
                ax_prob=ax_prob,
                tensor_prob=tensor_prob,
                par_prob=par_prob,
                single_formula=single_formula,
            )

            if not check(proof):
                raise RuntimeError("invalid proof")

            f.write(line_str(proof) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--out", type=Path, default=Path("mll_positional.txt"))
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--min-leaves", type=int, default=2)
    ap.add_argument("--max-leaves", type=int, default=16)
    ap.add_argument(
        "--n-atoms",
        type=int,
        default=10,
        help="atom name pool size; each atomic axiom draws uniformly in 0..n_atoms-1",
    )

    ap.add_argument(
        "--one-prob",
        type=float,
        default=0.20,
        help="probability of a 𝟙 leaf",
    )
    ap.add_argument(
        "--ax-prob",
        type=float,
        default=0.60,
        help="probability of an atomic axiom leaf; residual is ⊢ 𝟙, ⊥",
    )
    ap.add_argument(
        "--tensor-prob",
        type=float,
        default=0.45,
        help="probability of ⊗ (or stop when the forest has size 1)",
    )
    ap.add_argument(
        "--par-prob",
        type=float,
        default=0.35,
        help="probability of ⅋; residual is ⊥",
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
        tensor_prob=args.tensor_prob,
        par_prob=args.par_prob,
        single_formula=args.single_formula,
    )

    print(f"{args.n} proofs written to {args.out}")


if __name__ == "__main__":
    main()
