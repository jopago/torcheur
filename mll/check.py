from __future__ import annotations

from .formulas import Bottom, One, Par, dual
from .proofs import (
    Ax,
    BotRule,
    OneRule,
    ParRule,
    Proof,
    TensorRule,
    tensor_premises,
)


def check(proof: Proof) -> bool:
    if isinstance(proof, Ax):
        if len(proof.sequent) != 2:
            return False
        left_formula, right_formula = proof.sequent
        # Identity axiom: any dual pair, not necessarily atomic.
        return dual(left_formula) == right_formula

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
        except (ValueError, TypeError, IndexError):
            return False

        return (
            expected_left == proof.left.sequent
            and expected_right == proof.right.sequent
            and check(proof.left)
            and check(proof.right)
        )

    return False
