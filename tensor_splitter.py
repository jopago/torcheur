from collections.abc import Callable

import torch

from mll import proofs as P
from models.configs import StateActionConfig
from models.mll_transformers import StateActionTransformer
from tokenizer import FormulaTokenizer

TensorSplitter = Callable[[P.Sequent], tuple[int, tuple[int, ...]]]


def tensor_splitter_from_transformer(
    transformer: StateActionTransformer,
    tokenizer: FormulaTokenizer,
    config: StateActionConfig,
) -> TensorSplitter:
    return lambda sequent: _split_sequent_tensors_transformer(
        transformer, tokenizer, config, sequent
    )


def _split_sequent_tensors_transformer(
    transformer: StateActionTransformer,
    tokenizer: FormulaTokenizer,
    config: StateActionConfig,
    sequent: P.Sequent,
    device: str = "cpu",
) -> tuple[int, tuple[int, ...]]:
    """
    Use a StateActionTransformer to split tensors in a given sequent.
    :return: the index of the formula to split and the list of indices of formulas that go to the left
    """
    formula_tokens = [
        torch.tensor(
            tokenizer.encode(P.fstr(formula))[: config.max_formula_len],
            dtype=torch.long,
        )
        for formula in sequent
    ]

    N = len(formula_tokens)
    L = max(len(t) for t in formula_tokens)

    x = torch.full(
        (1, N, L),
        config.pad_token_id,
        dtype=torch.long,
        device=device,
    )

    for i, tokens in enumerate(formula_tokens):
        x[0, i, : len(tokens)] = tokens.to(device)

    with torch.no_grad():
        split_logits, side_logits = transformer(x)

    # Which tensor to split
    principal_index = split_logits[0].argmax().item()

    # Formulas assigned to LEFT
    side_pred = side_logits[0].argmax(dim=-1)

    left_positions = tuple(
        i for i in range(N) if i != principal_index and side_pred[i].item() == 1
    )

    return principal_index, left_positions
