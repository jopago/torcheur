import random
from collections.abc import Callable

import numpy as np
import torch

from mll import proofs as P
from mll.formulas import Tensor
from models.configs import StateActionConfig
from models.mll_transformers import (
    SideTransformer,
    SplitTransformer,
    StateActionTransformer,
)
from tokenizer import FormulaTokenizer

TensorSplitter = Callable[[P.Sequent], tuple[int, tuple[int, ...]]]


def random_tensor_split(sequent: P.Sequent) -> tuple[int, tuple[int, ...]]:
    if not sequent:
        return -1, ()

    tensor_idx = [i for i, formula in enumerate(sequent) if isinstance(formula, Tensor)]
    if not tensor_idx:
        return -1, ()

    split_index = int(np.random.choice(tensor_idx))
    left = tuple(
        i for i in range(len(sequent)) if i != split_index and np.random.randint(2)
    )
    return split_index, left


def _encode_sequent(
    sequent: P.Sequent,
    tokenizer: FormulaTokenizer,
    config: StateActionConfig,
    device: torch.device,
) -> torch.Tensor | None:
    """Pad sequent formulas to [1, N, L], or None if empty / unusable."""
    if not sequent:
        return None

    formula_tokens = [
        torch.tensor(
            tokenizer.encode(P.fstr(formula))[: config.max_formula_len],
            dtype=torch.long,
        )
        for formula in sequent
    ]

    N = len(formula_tokens)
    L = max((len(t) for t in formula_tokens), default=0)
    if L == 0:
        return None

    x = torch.full(
        (1, N, L),
        config.pad_token_id,
        dtype=torch.long,
        device=device,
    )
    for i, tokens in enumerate(formula_tokens):
        x[0, i, : len(tokens)] = tokens.to(device)
    return x


def _fallback_tensor_index(sequent: P.Sequent, principal_index: int) -> int:
    if isinstance(sequent[principal_index], Tensor):
        return principal_index
    tensor_indices = [
        i for i, formula in enumerate(sequent) if isinstance(formula, Tensor)
    ]
    if not tensor_indices:
        return -1
    return random.choice(tensor_indices)


def tensor_splitter_from_transformer(
    transformer: StateActionTransformer,
    tokenizer: FormulaTokenizer,
    config: StateActionConfig,
) -> TensorSplitter:
    return lambda sequent: _split_sequent_tensors_transformer(
        transformer, tokenizer, config, sequent
    )


def tensor_splitter_from_split_side(
    split_model: SplitTransformer,
    side_model: SideTransformer,
    tokenizer: FormulaTokenizer,
    config: StateActionConfig,
) -> TensorSplitter:
    return lambda sequent: _split_sequent_tensors_split_side(
        split_model, side_model, tokenizer, config, sequent
    )


def _split_sequent_tensors_transformer(
    transformer: StateActionTransformer,
    tokenizer: FormulaTokenizer,
    config: StateActionConfig,
    sequent: P.Sequent,
) -> tuple[int, tuple[int, ...]]:
    """
    Use a StateActionTransformer to split tensors in a given sequent.
    :return: the index of the formula to split and the list of indices of formulas that go to the left
    """
    device = next(transformer.parameters()).device
    x = _encode_sequent(sequent, tokenizer, config, device)
    if x is None:
        return -1, ()

    with torch.no_grad():
        split_logits, side_logits = transformer(x)

    principal_index = _fallback_tensor_index(sequent, split_logits[0].argmax().item())
    if principal_index < 0:
        return -1, ()

    side_pred = side_logits[0].argmax(dim=-1)
    left_positions = tuple(
        i
        for i in range(x.shape[1])
        if i != principal_index and side_pred[i].item() == 1
    )
    return principal_index, left_positions


def _split_sequent_tensors_split_side(
    split_model: SplitTransformer,
    side_model: SideTransformer,
    tokenizer: FormulaTokenizer,
    config: StateActionConfig,
    sequent: P.Sequent,
) -> tuple[int, tuple[int, ...]]:
    """Split with SplitTransformer, then sides with SideTransformer given that choice."""
    device = next(split_model.parameters()).device
    x = _encode_sequent(sequent, tokenizer, config, device)
    if x is None:
        return -1, ()

    with torch.no_grad():
        split_logits = split_model(x)
        principal_index = _fallback_tensor_index(
            sequent, split_logits[0].argmax().item()
        )
        if principal_index < 0:
            return -1, ()

        split_index = torch.tensor([principal_index], device=device)
        side_logits = side_model(x, split_index)
        side_pred = side_logits[0].argmax(dim=-1)

    left_positions = tuple(
        i
        for i in range(x.shape[1])
        if i != principal_index and side_pred[i].item() == 1
    )
    return principal_index, left_positions
