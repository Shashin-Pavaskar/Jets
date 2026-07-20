# Susceptibility Probes of Hierarchical Structure in Diffusion Models

This project probes the hierarchical structure of QCD jets using score-based
diffusion models. Multi-particle phase-space events generated via the SARGE
algorithm are subjected to a forward noising process and then denoised using a
pre-trained score network (PET) in q-space Langevin mode. Energy Flow
Polynomials (EFPs) computed on the clean and reconstructed samples measure the
correlation between original and reconstructed data as a function of noise
level; the resulting susceptibility probes the hierarchical structure of jet
phase space.

The same diagnostic is applied to two control domains: a single-class image
dataset and synthetic Gaussian data, where the
absence of hierarchy gives a null baseline.

## Method

1. **Event generation** — Multi-particle QCD antenna events generated with the
   SARGE algorithm.
2. **Forward process** — Events are noised to a range of levels
   t ∈ {0.1, 0.2, ..., 1.0} (as a fraction of the full diffusion time).
3. **Reverse process** — The pre-trained PET score network denoises each noised
   sample back to t = 0, repeated over N stochastic runs per noise level.
4. **Susceptibility** — EFPs are computed on the original and denoised samples.
   Their correlation as a function of t defines the susceptibility χ(t), whose
   structure reflects the hierarchy of the underlying data.

## Data and checkpoints

Trained SARGE checkpoints are from
[GenerativeModelsOnPhaseSpace](https://github.com/ibrahimEls/GenerativeModelsOnPhaseSpace)
(Bogorad, Elsharkawy, Kahn, Larkoski & Levi), the code accompanying
[arXiv:2604.02415](https://arxiv.org/abs/2604.02415). The same architecture was
retrained here on Gaussian data as a control.

## Setup

...

## Reproducing the results

...

## Citation

...
