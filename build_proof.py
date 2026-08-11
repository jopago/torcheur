from __future__ import annotations

from collections.abc import Callable

from mll import proofs as P
from mll.formulas import Atom, Bottom, One, Par, Tensor, dual

TensorSplitter = Callable[[P.Sequent], tuple[int, tuple[int, ...]]]


class ProofFromSplitterBuilder:
    def __init__(self, splitter: TensorSplitter):
        self.splitter = splitter

    def build_proof(self, sequent: P.Sequent) -> P.Proof:
        if not sequent:
            raise ValueError("empty sequent")

        return self._negative_phase(sequent, 0)

    def _negative_phase(
        self,
        sequent: P.Sequent,
        position: int,
    ) -> P.Proof:

        if position == len(sequent):
            if sequent == (One(),):
                return P.OneRule(sequent)

            if len(sequent) == 2 and sequent[0] == dual(sequent[1]):
                return P.Ax(sequent)

            return self._positive_phase(sequent)

        formula = sequent[position]

        if isinstance(formula, Par):
            child_sequent = (
                sequent[:position]
                + (formula.left, formula.right)
                + sequent[position + 1 :]
            )

            return P.ParRule(
                sequent,
                position,
                self._negative_phase(child_sequent, position),
            )

        if isinstance(formula, Bottom):
            child_sequent = sequent[:position] + sequent[position + 1 :]

            return P.BotRule(
                sequent,
                position,
                self._negative_phase(child_sequent, position),
            )

        return self._negative_phase(sequent, position + 1)

    def _negative_block_phase(
        self,
        sequent: P.Sequent,
        position: int,
        end_position: int,
    ) -> P.Proof:

        if position == end_position:
            if sequent == (One(),):
                return P.OneRule(sequent)

            if len(sequent) == 2 and sequent[0] == dual(sequent[1]):
                return P.Ax(sequent)

            return self._positive_phase(sequent)

        formula = sequent[position]

        if isinstance(formula, Par):
            child_sequent = (
                sequent[:position]
                + (formula.left, formula.right)
                + sequent[position + 1 :]
            )

            return P.ParRule(
                sequent,
                position,
                self._negative_block_phase(
                    child_sequent,
                    position,
                    end_position + 1,
                ),
            )

        if isinstance(formula, Bottom):
            child_sequent = sequent[:position] + sequent[position + 1 :]

            return P.BotRule(
                sequent,
                position,
                self._negative_block_phase(
                    child_sequent,
                    position,
                    end_position - 1,
                ),
            )

        return self._negative_block_phase(
            sequent,
            position + 1,
            end_position,
        )

    def _positive_phase(self, sequent: P.Sequent) -> P.Proof:

        if len(sequent) == 2 and all(isinstance(f, Atom) for f in sequent):
            if sequent[0] != dual(sequent[1]):
                raise ValueError("invalid atomic axiom")
            return P.Ax(sequent)

        if sequent == (One(),):
            return P.OneRule(sequent)

        principal_index, left_positions = self.splitter(sequent)

        if not (0 <= principal_index < len(sequent)):
            raise IndexError("principal_index out of range")

        tensor = sequent[principal_index]

        if not isinstance(tensor, Tensor):
            raise TypeError("principal formula is not a tensor")

        left_positions = tuple(left_positions)

        if left_positions != tuple(sorted(left_positions)):
            raise ValueError("left_positions must be sorted")

        if principal_index in left_positions:
            raise ValueError("principal index belongs to no branch")

        left_sequent, right_sequent = P.tensor_premises(
            sequent,
            principal_index,
            left_positions,
        )

        left_cursor = sum(1 for i in range(principal_index) if i in left_positions)
        right_cursor = principal_index - left_cursor

        if isinstance(tensor.left, (Par, Bottom)):
            left_proof = self._negative_block_phase(
                left_sequent,
                left_cursor,
                left_cursor + 1,
            )
        else:
            left_proof = self._positive_phase(left_sequent)

        if isinstance(tensor.right, (Par, Bottom)):
            right_proof = self._negative_block_phase(
                right_sequent,
                right_cursor,
                right_cursor + 1,
            )
        else:
            right_proof = self._positive_phase(right_sequent)

        return P.TensorRule(
            sequent,
            principal_index,
            left_positions,
            left_proof,
            right_proof,
        )
