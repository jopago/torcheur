from __future__ import annotations

from dataclasses import dataclass

from mll import proofs as P
from mll.formulas import Atom, Bottom, Formula, One, Par, Tensor, dual


@dataclass
class Block:
    """Negative expansion of one formula of the current original sequent.

    ``start`` is the block's (possibly lazy) start position in the normalized
    sequent.  It is meaningful only while the block is top-level.

    ``width`` is the number of visible formulas represented by the block after
    eager ⅋/⊥ decomposition.  It is computed once when the block's negative
    phase finishes and then cached.

    A leaf has ``left = right = None`` and width 1.
    An opened ⊥ has ``bot = True`` and width 0.
    An opened A ⅋ B has left/right children and
    ``width = left.width + right.width``.

    ``next_leaf`` is used only while constructing a block in a negative phase.
    """

    formula: Formula
    start: int = 0
    left: Block | None = None
    right: Block | None = None
    bot: bool = False
    width: int = 1
    next_leaf: Block | None = None


def normalize(proof: P.Proof) -> P.Proof:
    """Make every ⅋/⊥ eager while preserving the positive-rule skeleton."""
    if not proof.sequent:
        raise ValueError("empty sequent")

    blocks = [Block(formula) for formula in proof.sequent]
    blocks[0].start = 0
    return negative_phase(proof, proof.sequent, blocks, 0, blocks[0], 0)


def negative_phase(
    proof: P.Proof,
    sequent: P.Sequent,
    blocks: list[Block],
    block_index: int,
    current_node: Block | None,
    position: int,
) -> P.Proof:
    """Initial negative phase: normalize all blocks from left to right."""
    if block_index == len(blocks):
        return positive_phase(proof, sequent, blocks)

    if current_node is None:
        block = blocks[block_index]
        finish_block(block)
        block_index += 1

        if block_index == len(blocks):
            return positive_phase(proof, sequent, blocks)

        blocks[block_index].start = block.start + block.width
        return negative_phase(
            proof,
            sequent,
            blocks,
            block_index,
            blocks[block_index],
            position,
        )

    formula = sequent[position]

    if isinstance(formula, Par):
        old_next = current_node.next_leaf
        left = Block(formula.left)
        right = Block(formula.right)
        left.next_leaf = right
        right.next_leaf = old_next
        current_node.left = left
        current_node.right = right

        child_sequent = (
            sequent[:position] + (formula.left, formula.right) + sequent[position + 1 :]
        )
        return P.ParRule(
            sequent,
            position,
            negative_phase(proof, child_sequent, blocks, block_index, left, position),
        )

    if isinstance(formula, Bottom):
        next_node = current_node.next_leaf
        current_node.bot = True
        current_node.width = 0
        child_sequent = sequent[:position] + sequent[position + 1 :]
        return P.BotRule(
            sequent,
            position,
            negative_phase(
                proof, child_sequent, blocks, block_index, next_node, position
            ),
        )

    return negative_phase(
        proof,
        sequent,
        blocks,
        block_index,
        current_node.next_leaf,
        position + 1,
    )


def positive_phase(
    proof: P.Proof,
    sequent: P.Sequent,
    blocks: list[Block],
    lazy_from: int | None = None,
    lazy_shift: int = 0,
) -> P.Proof:
    """Skip negative rules already emitted and preserve the next positive rule.

    If ``lazy_from`` is not None, every top-level block whose index is at least
    ``lazy_from`` is shifted by ``lazy_shift`` in ``sequent``.  This is the only
    position change created by ``negative_block_phase``; it is materialized when
    the next tensor splits the context.
    """
    if isinstance(proof, P.ParRule):
        block = blocks[proof.index]
        if block.left is None or block.right is None:
            raise ValueError("Par principal was not opened in a negative phase")

        # Children inherit the parent's nominal (pre-lazy) coordinate system.
        block.left.start = block.start
        block.right.start = block.start + block.left.width
        blocks[proof.index : proof.index + 1] = [block.left, block.right]

        if lazy_from is not None and proof.index < lazy_from:
            lazy_from += 1

        return positive_phase(proof.child, sequent, blocks, lazy_from, lazy_shift)

    if isinstance(proof, P.BotRule):
        block = blocks[proof.index]
        if not block.bot:
            raise ValueError("Bottom principal was not removed in a negative phase")

        del blocks[proof.index]
        if lazy_from is not None and proof.index < lazy_from:
            lazy_from -= 1

        return positive_phase(proof.child, sequent, blocks, lazy_from, lazy_shift)

    if isinstance(proof, P.Ax):
        if len(sequent) == 2 and all(isinstance(f, Atom) for f in sequent):
            if sequent[0] != dual(sequent[1]):
                raise ValueError("invalid atomic axiom")
            return P.Ax(sequent)

        # The only non-atomic axiom case we accept is the original AX(⊥, 1)
        # (or AX(1, ⊥)); eager ⊥ has already reduced it to ⊢ 1.
        if sequent == (One(),):
            return P.OneRule(sequent)

        raise ValueError("axioms must be atomic, except for the ⊥/1 case")

    if isinstance(proof, P.OneRule):
        if sequent != (One(),):
            raise ValueError("invalid 1 rule after normalization")
        return P.OneRule(sequent)

    if not isinstance(proof, P.TensorRule):
        raise TypeError(type(proof))

    tensor = proof.sequent[proof.index]
    if not isinstance(tensor, Tensor):
        raise TypeError("TensorRule principal is not a tensor")

    def start(i: int) -> int:
        block_start = blocks[i].start
        if lazy_from is not None and i >= lazy_from:
            block_start += lazy_shift
        return block_start

    principal_block = blocks[proof.index]
    if principal_block.width != 1:
        raise ValueError("tensor principal does not occupy one visible position")

    principal_position = start(proof.index)

    # No prefix-sum scan: each original left block already knows where its
    # whole visible interval starts.
    left_positions: list[int] = []
    for i in proof.left_positions:
        block = blocks[i]
        block_start = start(i)
        left_positions.extend(range(block_start, block_start + block.width))

    left_positions_tuple = tuple(left_positions)
    left_sequent, right_sequent = P.tensor_premises(
        sequent, principal_position, left_positions_tuple
    )

    left_original = set(proof.left_positions)
    left_blocks: list[Block] = []
    right_blocks: list[Block] = []
    left_cursor = 0
    right_cursor = 0
    left_principal_index = -1
    right_principal_index = -1

    for i, block in enumerate(blocks):
        if i == proof.index:
            left_principal_index = len(left_blocks)
            right_principal_index = len(right_blocks)

            left_block = Block(tensor.left, start=left_cursor)
            right_block = Block(tensor.right, start=right_cursor)
            left_blocks.append(left_block)
            right_blocks.append(right_block)
            left_cursor += 1
            right_cursor += 1

        elif i in left_original:
            block.start = left_cursor
            left_blocks.append(block)
            left_cursor += block.width

        else:
            block.start = right_cursor
            right_blocks.append(block)
            right_cursor += block.width

    left_block = left_blocks[left_principal_index]
    if isinstance(tensor.left, (Par, Bottom)):
        left_proof = negative_block_phase(
            proof.left,
            left_sequent,
            left_blocks,
            left_principal_index,
            left_block,
            left_block.start,
            left_block.width,
        )
    else:
        left_proof = positive_phase(proof.left, left_sequent, left_blocks)

    right_block = right_blocks[right_principal_index]
    if isinstance(tensor.right, (Par, Bottom)):
        right_proof = negative_block_phase(
            proof.right,
            right_sequent,
            right_blocks,
            right_principal_index,
            right_block,
            right_block.start,
            right_block.width,
        )
    else:
        right_proof = positive_phase(proof.right, right_sequent, right_blocks)

    return P.TensorRule(
        sequent,
        principal_position,
        left_positions_tuple,
        left_proof,
        right_proof,
    )


def negative_block_phase(
    proof: P.Proof,
    sequent: P.Sequent,
    blocks: list[Block],
    block_index: int,
    current_node: Block | None,
    position: int,
    old_width: int,
) -> P.Proof:
    """Negative phase restricted to the single block created by a tensor."""
    block = blocks[block_index]

    if current_node is None:
        finish_block(block)
        shift = block.width - old_width
        lazy_from = block_index + 1 if shift else None
        return positive_phase(proof, sequent, blocks, lazy_from, shift)

    formula = sequent[position]

    if isinstance(formula, Par):
        old_next = current_node.next_leaf
        left = Block(formula.left)
        right = Block(formula.right)
        left.next_leaf = right
        right.next_leaf = old_next
        current_node.left = left
        current_node.right = right

        child_sequent = (
            sequent[:position] + (formula.left, formula.right) + sequent[position + 1 :]
        )
        return P.ParRule(
            sequent,
            position,
            negative_block_phase(
                proof,
                child_sequent,
                blocks,
                block_index,
                left,
                position,
                old_width,
            ),
        )

    if isinstance(formula, Bottom):
        next_node = current_node.next_leaf
        current_node.bot = True
        current_node.width = 0
        child_sequent = sequent[:position] + sequent[position + 1 :]
        return P.BotRule(
            sequent,
            position,
            negative_block_phase(
                proof,
                child_sequent,
                blocks,
                block_index,
                next_node,
                position,
                old_width,
            ),
        )

    return negative_block_phase(
        proof,
        sequent,
        blocks,
        block_index,
        current_node.next_leaf,
        position + 1,
        old_width,
    )


def finish_block(block: Block) -> int:
    """Compute and cache all subtree widths once, at the end of its negative phase."""
    if block.bot:
        block.width = 0
    elif block.left is not None and block.right is not None:
        block.width = finish_block(block.left) + finish_block(block.right)
    else:
        block.width = 1
    return block.width
