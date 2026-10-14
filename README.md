## Torcheur: learning-to-prove

**Torcheur** is an automated theorem prover for **Multiplicative Linear Logic (MLL)** and an experimental framework for learning proof 
search with neural networks.

The project automatically generates random MLL statements and valid proofs to create a synthetic dataset for transformer-based 
models implemented in *PyTorch*.

# Results

Full-proof success rate on 2000 validation sequents (same eval protocol):

| Splitter | Proof accuracy |
|----------|----------------|
| Random (uniform over ⊗ choices and LEFT/RIGHT partitions) | 27.9% |
| Joint `StateActionTransformer` (split + sides in parallel) | 52.25% |
| Conditional split → side (`SplitTransformer` then `SideTransformer`) | 58.7% |

# TODO:

- [x] automated generation of MLL statement and proofs
- [x] compact serialization of MLL proofs
- [x] transformer-based architecture for autoregressive proof generation
- [x] generate *(proof state, next action)* dataset
- [x] learn (state, action) next-step predictors (joint and split→side)
- [x] integrate the next-step predictor into proof construction and evaluate it
- [x] beat the random splitter baseline
- [ ] unit tests / proof visualization to sanity-check generated and predicted proofs
- [ ] stronger benchmark (OOD / harder sequents)
- [ ] atom-balancing deterministic baseline (+ optional constraint in the model)
