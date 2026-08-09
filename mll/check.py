from __future__ import annotations

from mll.formulas import Bottom, One, Par, dual
from mll.parse import (
    RawAx,
    RawBot,
    RawOne,
    RawPar,
    RawProof,
    RawTensor,
    try_parse,
)
from mll.proofs import Sequent, tensor_premises


def is_valid(sequent: Sequent, raw: RawProof) -> bool:
    """Whether raw is a correct MLL derivation of sequent."""
    if isinstance(raw, RawAx):
        if len(sequent) != 2:
            return False
        left_formula, right_formula = sequent
        return dual(left_formula) == right_formula

    if isinstance(raw, RawOne):
        return sequent == (One(),)

    if isinstance(raw, RawBot):
        index = raw.index
        if not (0 <= index < len(sequent)):
            return False
        if not isinstance(sequent[index], Bottom):
            return False
        child_sequent = sequent[:index] + sequent[index + 1 :]
        return is_valid(child_sequent, raw.child)

    if isinstance(raw, RawPar):
        index = raw.index
        if not (0 <= index < len(sequent)):
            return False
        principal = sequent[index]
        if not isinstance(principal, Par):
            return False
        child_sequent = (
            sequent[:index] + (principal.left, principal.right) + sequent[index + 1 :]
        )
        return is_valid(child_sequent, raw.child)

    if isinstance(raw, RawTensor):
        if not (0 <= raw.index < len(sequent)):
            return False
        try:
            left_sequent, right_sequent = tensor_premises(
                sequent,
                raw.index,
                raw.left_positions,
            )
        except (ValueError, TypeError, IndexError):
            return False
        return is_valid(left_sequent, raw.left) and is_valid(right_sequent, raw.right)

    return False


def check_syntax(line: str) -> bool:
    """True iff the line parses as a well-formed sequent || proof."""
    return try_parse(line) is not None


def check_proof(line: str) -> bool:
    """True iff the line parses and denotes a valid proof."""
    parsed = try_parse(line)
    if parsed is None:
        return False
    sequent, raw = parsed
    return is_valid(sequent, raw)


def check_all(line: str) -> tuple[bool, bool]:
    """Return (syntax_ok, proof_ok) with a single parse."""
    parsed = try_parse(line)
    if parsed is None:
        return False, False
    sequent, raw = parsed
    return True, is_valid(sequent, raw)
