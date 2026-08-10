from __future__ import annotations

import random
from pathlib import Path

from mll.formulas import Atom, Bottom, Formula, One, Par, Tensor, atom_name, dual
from mll.negative_normalization import normalize
from mll.proofs import (
    Ax,
    BotRule,
    OneRule,
    ParRule,
    Proof,
    TensorRule,
    proof_str,
    sstr,
)


def make_atom_ax(name: str, rng: random.Random, ax_pos_prob: float) -> Ax:
    atom = Atom(name)
    atom_dual = dual(atom)
    if rng.random() < ax_pos_prob:
        return Ax((atom, atom_dual))
    return Ax((atom_dual, atom))


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


def interleave(
    left_formulas: tuple[Formula, ...] | list[Formula],
    right_formulas: tuple[Formula, ...] | list[Formula],
    rng: random.Random,
) -> tuple[list[Formula], list[str]]:
    """Shuffle two contexts while preserving each side's relative order."""
    merged: list[Formula] = []
    origins: list[str] = []
    left_index = 0
    right_index = 0

    while left_index < len(left_formulas) or right_index < len(right_formulas):
        if left_index >= len(left_formulas):
            take_left = False
        elif right_index >= len(right_formulas):
            take_left = True
        else:
            take_left = bool(rng.getrandbits(1))

        if take_left:
            merged.append(left_formulas[left_index])
            origins.append("L")
            left_index += 1
        else:
            merged.append(right_formulas[right_index])
            origins.append("R")
            right_index += 1

    return merged, origins


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

    before, before_origins = interleave(gamma1, delta1, rng)
    after, after_origins = interleave(gamma2, delta2, rng)

    sequent = tuple(before + [Tensor(left_factor, right_factor)] + after)
    principal_index = len(before)
    origins = before_origins + ["T"] + after_origins

    left_positions = tuple(
        position for position, origin in enumerate(origins) if origin == "L"
    )

    return TensorRule(
        sequent=sequent,
        index=principal_index,
        left_positions=left_positions,
        left=left,
        right=right,
    )


def weighted_choice(rng: random.Random, weights: dict[str, float]) -> str:
    items = [(name, weight) for name, weight in weights.items() if weight > 0]
    if not items:
        raise ValueError("no positive weights")
    total = sum(weight for _, weight in items)
    roll = rng.random() * total
    cumulative = 0.0
    for name, weight in items:
        cumulative += weight
        if roll < cumulative:
            return name
    return items[-1][0]


def generate_proof(
    rng: random.Random,
    n_leaves: int,
    n_atoms: int,
    one_prob: float,
    ax_prob: float,
    ax_pos_prob: float,
    tensor_prob: float,
    par_prob: float,
    stop_prob: float,
    single_formula: bool,
) -> Proof:
    if n_atoms < 1:
        raise ValueError("n_atoms must be >= 1")
    if one_prob < 0 or ax_prob < 0 or one_prob + ax_prob > 1:
        raise ValueError("need one_prob >= 0, ax_prob >= 0, one_prob + ax_prob <= 1")
    if not (0 <= ax_pos_prob <= 1):
        raise ValueError("ax_pos_prob must be in [0, 1]")
    if tensor_prob < 0 or par_prob < 0 or tensor_prob + par_prob > 1:
        raise ValueError(
            "need tensor_prob >= 0, par_prob >= 0, tensor_prob + par_prob <= 1"
        )
    if not (0 <= stop_prob <= 1):
        raise ValueError("stop_prob must be in [0, 1]")

    bot_prob = 1.0 - tensor_prob - par_prob
    forest: list[Proof] = []

    for _ in range(n_leaves):
        roll = rng.random()
        if roll < one_prob:
            forest.append(make_one())
        elif roll < one_prob + ax_prob:
            forest.append(
                make_atom_ax(atom_name(rng.randrange(n_atoms)), rng, ax_pos_prob)
            )
        else:
            forest.append(make_unit_ax())

    while True:
        par_indices = [i for i, proof in enumerate(forest) if len(proof.sequent) >= 2]
        can_par = bool(par_indices)

        if len(forest) > 1:
            if can_par:
                action = weighted_choice(
                    rng,
                    {"tensor": tensor_prob, "par": par_prob, "bot": bot_prob},
                )
            else:
                # Unavailable ⅋ mass is transferred to ⊗, not to ⊥.
                action = weighted_choice(
                    rng,
                    {"tensor": tensor_prob + par_prob, "bot": bot_prob},
                )

            if action == "tensor":
                left_index = rng.randrange(len(forest))
                left = forest.pop(left_index)
                right_index = rng.randrange(len(forest))
                right = forest.pop(right_index)
                forest.append(make_tensor(left, right, rng))
            elif action == "par":
                index = rng.choice(par_indices)
                forest[index] = make_par(forest[index], rng)
            else:
                index = rng.randrange(len(forest))
                forest[index] = make_bot(forest[index], rng)
            continue

        # Forest size 1: ⊗ is impossible; stop_prob controls termination.
        if rng.random() < stop_prob:
            proof = forest[0]
            if single_formula:
                while len(proof.sequent) > 1:
                    proof = make_par(proof, rng)
            return proof

        if can_par and (par_prob > 0 or bot_prob > 0):
            action = weighted_choice(rng, {"par": par_prob, "bot": bot_prob})
            if action == "par":
                forest[0] = make_par(forest[0], rng)
            else:
                forest[0] = make_bot(forest[0], rng)
        else:
            forest[0] = make_bot(forest[0], rng)


def generate_dataset(
    out: Path,
    n: int,
    seed: int,
    min_leaves: int,
    max_leaves: int,
    n_atoms: int,
    one_prob: float,
    ax_prob: float,
    ax_pos_prob: float,
    tensor_prob: float,
    par_prob: float,
    stop_prob: float,
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
                ax_pos_prob=ax_pos_prob,
                tensor_prob=tensor_prob,
                par_prob=par_prob,
                stop_prob=stop_prob,
                single_formula=single_formula,
            )
            f.write(f"{sstr(proof.sequent)} || {proof_str(proof)}.\n")


def iter_tensor_lines(proof: Proof) -> list[str]:
    """One dataset line per ⊗ rule: sequent || index, [goes_left...]."""
    lines: list[str] = []

    if isinstance(proof, (BotRule, ParRule)):
        lines.extend(iter_tensor_lines(proof.child))
    elif isinstance(proof, TensorRule):
        left_set = set(proof.left_positions)
        main_index = proof.index
        encoded = []
        for i in range(len(proof.sequent)):
            if i == main_index:
                encoded.append(2)
            elif i in left_set:
                encoded.append(1)
            else:
                encoded.append(0)
        lines.append(f"{sstr(proof.sequent)} || {encoded}.")
        lines.extend(iter_tensor_lines(proof.left))
        lines.extend(iter_tensor_lines(proof.right))

    return lines


def generate_normalized_dataset(
    out: Path,
    n: int,
    seed: int,
    min_leaves: int,
    max_leaves: int,
    n_atoms: int,
    one_prob: float,
    ax_prob: float,
    ax_pos_prob: float,
    tensor_prob: float,
    par_prob: float,
    stop_prob: float,
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
                ax_pos_prob=ax_pos_prob,
                tensor_prob=tensor_prob,
                par_prob=par_prob,
                stop_prob=stop_prob,
                single_formula=single_formula,
            )
            normalized = normalize(proof)
            for line in iter_tensor_lines(normalized):
                f.write(f"{line}\n")
