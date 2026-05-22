# Jet Phase Space Diffusion

## Overview
This project probes the hierarchical structure of QCD jets using score-based diffusion models. 
Multi-particle phase space events generated via the SARGE algorithm are subjected to a 
forward noising process and then denoised using a pre-trained Langevin score network (PET). 
Energy Flow Polynomials (EFPs) computed on the clean and reconstructed samples are used to 
measure susceptibility as a function of the noise level — providing a probe of the underlying 
hierarchical structure of jet phase space.

## Method
1. **Event Generation** — Multi-particle QCD antenna events are generated using the SARGE 
   algorithm in q-space (Lorentz-invariant phase space representation)
2. **Forward Process** — Events are noised to varying levels T★ ∈ {T/10, 2T/10, ..., T} 
   using Langevin dynamics with a physics-informed score function derived from the 
   Lorentz-invariant phase space prior
3. **Reverse Process** — The pre-trained PET score network denoises each noised sample 
   back to t=0, repeated across multiple stochastic runs per noise level
4. **Susceptibility** — EFPs are computed on original and denoised samples; their 
   correlations as a function of T★ probe the susceptibility of jet structure to noise. 
