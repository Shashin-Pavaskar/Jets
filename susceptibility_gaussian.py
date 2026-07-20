"""
Susceptibility analysis for Gaussian null test in q-space.
Adapted from denoise_qspace_xi20.ipynb.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ── Paths ─────────────────────────────────────────────────────────────────────
CKPT_PATH   = (
    '/projects/illinois/eng/physics/yfkahn/GenerativeModelsOnPhaseSpace/TrainedModels'
    '/_qspace_small_dsm_eps_N10_T1000_gaussian_N10_100K_pavaskar/v5_qspace_small_dsm_eps_N10_T1000_gaussian_N10_100K_pavaskar/checkpoints/_qspace_small_dsm_eps_N10_T1000_gaussian_N10_100K_pavaskar-epoch=000595-step=056024-val_loss=0.6518.ckpt'
)
INPUT_PATH  = '/scratch/pavaskar/gaussian_N10_100K.pt'
SAVE_PATH   = '/scratch/pavaskar/denoised_checkpoint_gaussian_595.pt'
OUTPUT_PATH = '/scratch/pavaskar/susceptibility_gaussian_595.pt'

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {DEVICE}')

# ── Imports from repo ─────────────────────────────────────────────────────────
import sys
sys.path.insert(0, '/projects/illinois/eng/physics/yfkahn/GenerativeModelsOnPhaseSpace')

from omnilearn_lightning.model import PETLightning
from omnilearn_lightning.diffusion import forward_step, reverse_step

# ── 1. Load model and schedule ─────────────────────────────────────────────────
print('Loading model...')
model = PETLightning.load_from_checkpoint(CKPT_PATH, map_location=DEVICE)
model.eval().to(DEVICE)

gammas = model.gammas.to(DEVICE).float()   # (T,)
T      = len(gammas)
t_gaus = model.t_gaus

print(f'T={T} | t_gaus={t_gaus}')
print(f'gamma: [{gammas.min():.6f}, {gammas.max():.6f}]')
print(f'Score regime: t in [{t_gaus}, {T-1}]')
print(f'Gaussian regime: t in [0, {t_gaus-1}]')

# ── 2. Load input data ─────────────────────────────────────────────────────────
print(f'\nLoading data from {INPUT_PATH}')
Q0_full = torch.load(INPUT_PATH, map_location='cpu').float()
print(f'Dataset shape: {Q0_full.shape}')

# ── 3. Setup ───────────────────────────────────────────────────────────────────
N_SAMPLES    = 4000
Q0           = Q0_full[:N_SAMPLES].to(DEVICE)
print(f'Working subset: {Q0.shape}')

# T_STAR grid: evenly spaced fractions of T
T_STARS      = [T//100, 3*T//100, 5*T//100, 7*T//100,9*T//100,  T//10, 2*T//10, 3*T//10, 4*T//10, 5*T//10, 6*T//10, 7*T//10, 8*T//10 ,9*T//10, T]
n_diffusion  = 10
BATCH_SIZE   = 1000

print(f'T_STARS: {T_STARS}')
print(f'n_diffusion: {n_diffusion}')

# ── 4. Forward process ─────────────────────────────────────────────────────────
print('\nRunning forward process...')
Q_noised_all = torch.zeros(len(T_STARS), n_diffusion, N_SAMPLES, 10, 3)

with torch.no_grad():
    for i, T_STAR in enumerate(T_STARS):
        for diff in range(n_diffusion):
            Q_noised = Q0.clone()
            for t in tqdm(range(T_STAR), desc=f'T_STAR={T_STAR} diff={diff+1}', leave=False):
                noscore = (t < t_gaus)
                Q_noised, _ = forward_step(Q_noised, gammas[t], noscore=noscore)
            Q_noised_all[i][diff] = Q_noised.cpu()

print(f'Q_noised_all shape: {Q_noised_all.shape}')
torch.save(Q_noised_all, '/scratch/pavaskar/Q_noised_gaussian.pt')
print('Saved Q_noised_all')

# ── 5. Reverse process ─────────────────────────────────────────────────────────

@torch.no_grad()
def reverse_from_tstar(Q_init, t_star, model, gammas, t_gaus, device, batch_size=250):
    N          = Q_init.shape[0]
    T_total    = len(gammas)
    gammas_rev = gammas.flip(0)
    s_start    = T_total - 1 - t_star
    score_fn   = lambda Q, t: model.model(Q, t)
    results    = []

    for start in range(0, N, batch_size):
        Q = Q_init[start : start + batch_size].clone().to(device)
        B = Q.shape[0]

        for s in tqdm(
            range(s_start, T_total),
            desc=f'Reverse batch {start//batch_size+1}/{(N-1)//batch_size+1}',
            leave=False,
        ):
            t_norm  = torch.full((B,), (T_total - s) / T_total,
                                 device=device, dtype=torch.float32)
            gamma_s = gammas_rev[s]
            noscore = (s >= T_total - t_gaus)

            Q = reverse_step(Q, t_norm, gamma_s, score_fn,
                             noscore=noscore, predict_eps=False)

        results.append(Q.cpu())

    return torch.cat(results, dim=0)


print('\nRunning reverse process...')
if os.path.exists(SAVE_PATH):
    saved      = torch.load(SAVE_PATH, map_location='cpu')
    Q_denoised = saved['Q_denoised']
    done       = saved['done']
    print(f'Resuming — {done.sum().item()} / {len(T_STARS) * n_diffusion} runs done')
else:
    Q_denoised = torch.zeros(len(T_STARS), n_diffusion, N_SAMPLES, 10, 3)
    done       = torch.zeros(len(T_STARS), n_diffusion, dtype=torch.bool)
    print('Starting fresh')

for i, T_STAR in enumerate(T_STARS):
    for diff in range(n_diffusion):
        if done[i, diff]:
            continue

        print(f'Running T_STAR={T_STAR} ({i+1}/{len(T_STARS)})  diff={diff+1}/{n_diffusion} ...', end=' ')
        Q_denoised[i, diff] = reverse_from_tstar(
            Q_init=Q_noised_all[i, diff],
            t_star=T_STAR,
            model=model,
            gammas=gammas,
            t_gaus=t_gaus,
            device=DEVICE,
            batch_size=BATCH_SIZE,
        )
        done[i, diff] = True
        torch.save({'Q_denoised': Q_denoised, 'done': done}, SAVE_PATH)
        print(f'done  [{done.sum().item()}/{len(T_STARS)*n_diffusion}]')

print(f'\nAll runs complete!')
print(f'Q_denoised shape: {Q_denoised.shape}')

# ── 6. Compute susceptibility χ(t) ────────────────────────────────────────────
print('\nComputing susceptibility...')

# χ(t*) = variance of denoised output across diffusion runs
# shape: (len(T_STARS), n_diffusion, N_SAMPLES, 10, 3)
# mean over diffusion runs, then variance over samples

# clean_dataset: original Gaussian q-space events (n_events, 10, 3)
clean_dataset = Q0_full[:N_SAMPLES].numpy()   # your x_0

# Tile to match denoised shape (T, n_diff, n_events, 10, 3)
results_new = np.tile(
    clean_dataset[np.newaxis, np.newaxis, :, :, :],   # (1, 1, n_events, 10, 3)
    (len(T_STARS), n_diffusion, 1, 1, 1)                              # (T, n_diff, n_events, 10, 3)
)

# Q_denoised is your denoised q-space data (T, n_diff, n_events, 10, 3)
diff = results_new - Q_denoised.numpy()              # x_0 - x_hat_0

# Correlator: contract over the 3 q-space components → (..., 10, 10)
correlator = np.einsum('...ib,...jb->...ij', diff, diff)

diff_avg = np.nanmean(diff, axis=1)                  # avg over diffusion trajectories
correlator_avg = np.nanmean(correlator, axis=1) - np.einsum('...ib,...jb->...ij', diff_avg, diff_avg)
corr_final = np.nanmean(correlator_avg, axis=1)      # avg over events

diag_sum = np.trace(corr_final , axis1=1, axis2=2)    # trace
diag_squared_sum = np.trace(corr_final@corr_final, axis1=1, axis2=2) 
total_sum = corr_final.sum(axis=(1, 2))              # full sum
chi = diag_squared_sum / diag_sum                           # susceptibility

print('chi(t):', chi)

# ── 7. Plot ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(np.array(T_STARS)/ T, chi, 'o-', color='steelblue', linewidth=2, markersize=6)
ax.set_xlabel('t* / T', fontsize=13)
ax.set_ylabel(r'$\chi(t^*)$', fontsize=13)
ax.set_title('Susceptibility — Gaussian null test', fontsize=13)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('/scratch/pavaskar/susceptibility_gaussian.png', dpi=150)
print('Saved plot to /scratch/pavaskar/susceptibility_gaussian.png')

# Save results
torch.save({'T_STARS': T_STARS, 'chi': chi, 'T': T}, OUTPUT_PATH)
print(f'Saved susceptibility data to {OUTPUT_PATH}')


eigenvalues = np.array([np.linalg.eigvalsh(C) for C in corr_final])  # (n_eps, d)

plt.figure(figsize=(7, 4))
for k in range(eigenvalues.shape[1]):
    plt.plot(np.array(T_STARS)/ T, eigenvalues[:, k], alpha=0.4, marker='o', markersize=3)
plt.xlabel('$\\varepsilon$')
plt.ylabel('$\\lambda_k$')
plt.title('All eigenvalues vs $\\varepsilon$')
plt.savefig('/scratch/pavaskar/gaussian_model_eigenvalues.png',dpi=150)
plt.tight_layout()
print('Saved plot to /scratch/pavaskar/gaussian_model_eigenvalues.png')

result = np.zeros(len(T_STARS))
result_std = np.zeros(len(T_STARS))

for t in range(len(T_STARS)):
    diff_t = diff[t]                      # (n_diff, n_events, 10, 3)
    
    # 1. Sum over the 10 particles
    summed=np.linalg.norm(diff_t, axis=-1)
    #summed = diff_t.sum(axis=2)           # (n_diff, n_events, 3)
    
    # 2. Take the norm of the summed 3-vector
    norms=summed.sum(axis=-1)
    #norms = np.linalg.norm(summed, axis=-1)  # (n_diff, n_events)
    
    # 3. Drop NaN values (from bad denoising) and average
    valid = norms[~np.isnan(norms)]       # 1D array of clean values
    
    result[t]     = valid.mean()
    result_std[t] = valid.std()
   # print(f'T★={T_STARS[t]:.3f}: kept {len(valid)}/{norms.size}  mean={result[t]:.4e}')

# Plot
plt.errorbar(T_STARS, result, marker='o', capsize=3)
plt.xlabel('t/T')
plt.ylabel(' Σ_i |s_i|')
plt.title('Absolute Magnetization vs noise level')
plt.savefig('/scratch/pavaskar/Absolute magnetization vs noise.png')
