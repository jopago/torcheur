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
from mll.proofs import (
    Ax,
    BotRule,
    OneRule,
    ParRule,
    Proof,
    Sequent,
    TensorRule,
    tensor_premises,
)


def check_proof(proof: Proof) -> bool:
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


def check_raw_proof(sequent: Sequent, raw: RawProof) -> bool:
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
        return check_raw_proof(child_sequent, raw.child)

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
        return check_raw_proof(child_sequent, raw.child)

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
        return check_raw_proof(left_sequent, raw.left) and check_raw_proof(
            right_sequent, raw.right
        )

    return False


def check_syntax(line: str) -> bool:
    """True iff the line parses as a well-formed sequent || proof."""
    return try_parse(line) is not None


def check_proof_of_line(line: str) -> bool:
    """True iff the line parses and denotes a valid proof."""
    parsed = try_parse(line)
    if parsed is None:
        return False
    sequent, raw = parsed
    return check_raw_proof(sequent, raw)


def check_all(line: str) -> tuple[bool, bool]:
    """Return (syntax_ok, proof_ok) with a single parse."""
    parsed = try_parse(line)
    if parsed is None:
        return False, False
    sequent, raw = parsed
    return True, check_raw_proof(sequent, raw)
