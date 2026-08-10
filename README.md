## Torcheur: learning-to-prove

**Torcheur** is an automated theorem prover for **Multiplicative Linear Logic (MLL)** and an experimental framework for learning proof 
search with neural networks.

The project automatically generates random MLL statements and valid proofs to create a synthetic dataset for transformer-based 
models implemented in *PyTorch*.

# TODO:

- [x] automated generation of MLL statement and proofs
- [x] compact serialization of MLL proofs
- [x] transformer-based architecture for autoregressive proof generation
- [ ] generate *proof state, next action)* dataset
- [ ] learn (state, action) proof
- [ ] integrate the next-step predictor into a proof search algorithm and evaluate it 