from __future__ import annotations

from mll import proofs as P
from mll.formulas import Atom, Bottom, One, Par, Tensor, dual
from mll.negative_normalization import Block


def build_proof(sequent: P.Sequent) -> P.Proof:
    """Build a focused-style proof of ``sequent`` (eager ⅋/⊥, AI-chosen ⊗)."""
    if not sequent:
        raise ValueError("empty sequent")

    blocks = [Block(formula) for formula in sequent]
    blocks[0].start = 0
    return negative_phase(sequent, blocks, 0, blocks[0], 0)


def choose_tensor_split(sequent: P.Sequent) -> tuple[int, tuple[int, ...]]:
    """Hole for the model: pick the principal ⊗ and which formulas go left.

    Must return
    -----------
    principal_index
        Index of the principal ``Tensor`` in ``sequent``.
    left_positions
        Sorted indices of formulas that go to the left premise.
        Must not include ``principal_index``.
    """
    raise NotImplementedError("to be implemented by jopago")


def negative_phase(
    sequent: P.Sequent,
    position: int,
) -> P.Proof:

    if position == len(sequent):
        if sequent == (One(),):
            return P.OneRule(sequent)

        if len(sequent) == 2 and sequent[0] == dual(sequent[1]):
            return P.Ax(sequent)

        return ai_positive_phase(sequent)

    formula = sequent[position]

    if isinstance(formula, Par):
        child_sequent = (
            sequent[:position] + (formula.left, formula.right) + sequent[position + 1 :]
        )
        return P.ParRule(sequent, position, negative_phase(child_sequent, position))

    if isinstance(formula, Bottom):
        child_sequent = sequent[:position] + sequent[position + 1 :]
        return P.BotRule(
            sequent,
            position,
            negative_phase(child_sequent, position),
        )

    return negative_phase(sequent, position + 1)


def negative_block_phase(
    sequent: P.Sequent, position: int, end_position: int
) -> P.Proof:

    if position == end_position:
        if sequent == (One(),):
            return P.OneRule(sequent)

        if len(sequent) == 2 and sequent[0] == dual(sequent[1]):
            return P.Ax(sequent)

        return ai_positive_phase(sequent)

    formula = sequent[position]

    if isinstance(formula, Par):
        child_sequent = (
            sequent[:position] + (formula.left, formula.right) + sequent[position + 1 :]
        )
        return P.ParRule(
            sequent,
            position,
            negative_block_phase(child_sequent, position, end_position + 1),
        )

    if isinstance(formula, Bottom):
        child_sequent = sequent[:position] + sequent[position + 1 :]
        return P.BotRule(
            sequent,
            position,
            negative_block_phase(child_sequent, position, end_position - 1),
        )

    return negative_block_phase(
        sequent,
        position + 1,
        end_position,
    )


def ai_positive_phase(sequent: P.Sequent) -> P.Proof:
    if len(sequent) == 2 and all(isinstance(f, Atom) for f in sequent):
        if sequent[0] != dual(sequent[1]):
            raise ValueError("invalid atomic axiom")
        return P.Ax(sequent)

    if sequent == (One(),):
        return P.OneRule(sequent)

    principal_index, left_positions = choose_tensor_split(sequent)

    if not (0 <= principal_index < len(sequent)):
        raise IndexError("principal_index out of range")
    tensor = sequent[principal_index]
    if not isinstance(tensor, Tensor):
        raise TypeError("principal formula is not a tensor")

    left_positions_tuple = tuple(sorted(left_positions))
    if left_positions_tuple != tuple(left_positions):
        raise ValueError("left_positions must be sorted")
    if principal_index in left_positions_tuple:
        raise ValueError("principal index belongs to no branch")

    left_sequent, right_sequent = P.tensor_premises(
        sequent, principal_index, left_positions_tuple
    )

    left_cursor = sum(1 for i in range(principal_index) if i in left_positions_tuple)
    right_cursor = principal_index - left_cursor

    if isinstance(tensor.left, (Par, Bottom)):
        left_proof = negative_block_phase(
            left_sequent, position=left_cursor, end_position=left_cursor + 1
        )
    else:
        left_proof = ai_positive_phase(left_sequent)

    if isinstance(tensor.right, (Par, Bottom)):
        right_proof = negative_block_phase(
            right_sequent, position=right_cursor, end_position=right_cursor + 1
        )
    else:
        right_proof = ai_positive_phase(right_sequent)

    return P.TensorRule(
        sequent,
        principal_index,
        left_positions_tuple,
        left_proof,
        right_proof,
    )
