# Susceptibility Probes of Hierarchical Structure in Diffusion Models

This project investigates whether statistical-physics-inspired response measures can be used to probe learned structure in generative diffusion models. We study score-based diffusion models trained on QCD jet data, where the underlying data contain nontrivial hierarchical correlations, and examine how these correlations are lost and reconstructed across different noise scales.

Multi-particle phase-space events generated with PYTHIA and SARGE are subjected to a forward noising process and then denoised using a pre-trained score network (PET) in q-space Langevin mode. Correlations between the original and reconstructed data are measured as a function of noise level and used to construct a susceptibility-like observable. For both PYTHIA and SARGE datasets, we observe a peak in this response as a function of noise scale.

As a control, we apply the same diagnostic to synthetic Gaussian data, which lack the hierarchical structure of the jet datasets. The absence of a comparable response provides a null baseline for studying the relationship between susceptibility and structure in the underlying data.

## Motivation

In statistical physics, response functions and susceptibilities provide macroscopic probes of changes in the organization of complex systems. This project explores whether analogous quantities can be useful for understanding how generative neural networks learn and reconstruct structured data.

The broader goal is to investigate statistical-physics-inspired diagnostics of learned structure and emergent behavior in machine-learning systems.

## Method

1. **Event generation** — Multi-particle QCD events are generated using PYTHIA and the SARGE algorithm.

2. **Forward process** — Events are noised to a range of levels \(t \in \{0.1, 0.2, \ldots, 1.0\}\), expressed as fractions of the full diffusion time.

3. **Reverse process** — A pre-trained PET score network denoises each noised sample back to \(t=0\), with multiple stochastic reverse trajectories generated at each noise level.

4. **Susceptibility** — Correlations between the original and reconstructed samples are measured as a function of noise level. Their response across the diffusion trajectory defines a susceptibility-like observable \(\chi(t)\), which is used to investigate the hierarchical structure of the underlying data.

## Data and Checkpoints

The architecture used for training is based on [GenerativeModelsOnPhaseSpace](https://github.com/ibrahimEls/GenerativeModelsOnPhaseSpace) (Bogorad, Elsharkawy, Kahn, Larkoski & Levi), the code accompanying [arXiv:2604.02415](https://arxiv.org/abs/2604.02415).

The same architecture was retrained on synthetic Gaussian data to provide a control dataset.
