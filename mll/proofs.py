from __future__ import annotations

from dataclasses import dataclass

from .formulas import Formula, Tensor, fstr

Sequent = tuple[Formula, ...]


def sstr(seq: Sequent) -> str:
    return "⊢ " + ", ".join(fstr(x) for x in seq)


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


def tensor_premises(
    seq: Sequent,
    principal_index: int,
    left_positions: tuple[int, ...],
) -> tuple[Sequent, Sequent]:
    """Rebuild the two ⊗ premises from the conclusion and left_positions."""
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


Proof = Ax | OneRule | BotRule | ParRule | TensorRule


def proof_str(proof: Proof) -> str:
    if isinstance(proof, Ax):
        return "AX"
    if isinstance(proof, OneRule):
        return "𝟙"
    if isinstance(proof, BotRule):
        return f"⊥({proof.index},{proof_str(proof.child)})"
    if isinstance(proof, ParRule):
        return f"⅋({proof.index},{proof_str(proof.child)})"
    if isinstance(proof, TensorRule):
        positions = "[" + ",".join(map(str, proof.left_positions)) + "]"
        return (
            f"⊗({proof.index},{positions},"
            f"{proof_str(proof.left)},{proof_str(proof.right)})"
        )
    raise TypeError(type(proof))
