"""
Image Diffusion Susceptibility -- Sclocchi et al. (arXiv:2410.13770) style analysis.

Pipeline:
  1. Load pre-trained pixel-space DDPM (Hugging Face diffusers)
  2. Forward-noise images to t*, backward-denoise to 0
  3. CLIP ViT-B/32 patch tokens (7x7 grid) for original + regenerated images
  4. Connected correlator of patch-embedding-norm variations
  5. Susceptibility chi(t) = sum_ij C_ij / Tr(C), spatial correlation length

Checkpointed: safe to resubmit after SLURM timeout, resumes automatically.

Usage:
  # once, on the LOGIN node (internet access), to cache models:
  python image_susceptibility.py --download-only

  # then submit the SLURM job (compute nodes run offline from cache):
  sbatch submit_image_susceptibility.sbatch
"""

import argparse
import glob
import os

import numpy as np
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------- config
#MODEL_ID   = 'google/ddpm-cifar10-32'
CLIP_ID    = "openai/clip-vit-base-patch16"

N_IMAGES   = 4000
BATCH      = 1024            # A100 80GB handles this comfortably at 256px
N_RUNS     = 5
T_FRACS    = [0.05, 0.1, 0.25, 0.4, 0.55, 0.7, 0.9]
GRID       = 14             # ViT-B/32 @ 224px -> 7x7 patches
N_PATCH    = GRID * GRID

MODEL_DIR  = "ddpm_celeba_bedroom_128"          # set to a folder of images, or None to sample from the DDPM
X0_PATH    = "ddpm_celeba_bedroom_128/x0_holdout.pt"     # use the REAL held-out images
SAVE_PATH  = "celeba_bedroom_susceptibility_checkpoint_128.npz"
OUT_PREFIX = "results_celeba_bedroom_128"
IMAGE_DIR  = None    # leave None — x0 comes from x0_holdout.pt

# ---------------------------------------------------------------- args
parser = argparse.ArgumentParser()
parser.add_argument("--download-only", action="store_true",
                    help="download/cache models on the login node, then exit")
args = parser.parse_args()

if args.download_only:
  #  from diffusers import UNet2DModel, DDPMScheduler
    from transformers import CLIPVisionModel
  #  UNet2DModel.from_pretrained(MODEL_ID)
  #  DDPMScheduler.from_pretrained(MODEL_ID)
    CLIPVisionModel.from_pretrained(CLIP_ID)
    print("CLIP cached -- ready for offline compute nodes.")
    raise SystemExit(0)

# ---------------------------------------------------------------- setup
from diffusers import UNet2DModel, DDPMScheduler          # noqa: E402
from transformers import CLIPVisionModel                    # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")

unet = UNet2DModel.from_pretrained(f"{MODEL_DIR}/unet").to(DEVICE).eval()
scheduler = DDPMScheduler.from_pretrained(f"{MODEL_DIR}/scheduler")
T_TOTAL    = scheduler.config.num_train_timesteps
alphas_bar = scheduler.alphas_cumprod.to(DEVICE)
IMG_SIZE   = unet.config.sample_size
T_STARS    = [int(f * (T_TOTAL - 1)) for f in T_FRACS]
print(f"T_total={T_TOTAL} | img={IMG_SIZE}px | t* = {T_STARS}")

clip = CLIPVisionModel.from_pretrained(CLIP_ID).to(DEVICE).eval()
CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073],
                         device=DEVICE).view(1, 3, 1, 1)
CLIP_STD  = torch.tensor([0.26862954, 0.26130258, 0.27577711],
                         device=DEVICE).view(1, 3, 1, 1)


# ---------------------------------------------------------------- helpers
@torch.no_grad()
def forward_noise(x0_batch, t_star):
    a = alphas_bar[t_star]
    eps = torch.randn_like(x0_batch)
    return a.sqrt() * x0_batch + (1 - a).sqrt() * eps


@torch.no_grad()
def backward_denoise(x_t, t_star):
    x = x_t.clone()
    scheduler.set_timesteps(T_TOTAL)
    steps = [t for t in scheduler.timesteps if t <= t_star]
    for t in steps:
        eps = unet(x, t).sample
        x = scheduler.step(eps, t, x).prev_sample
    return x


@torch.no_grad()
def forward_backward(x0_all, t_star, batch=BATCH):
    outs = []
    for s in range(0, len(x0_all), batch):
        xb = x0_all[s:s + batch].to(DEVICE)
        xt = forward_noise(xb, t_star)
        outs.append(backward_denoise(xt, t_star).cpu())
    return torch.cat(outs)


@torch.no_grad()
def patch_embeddings(images, batch=BATCH):
    """images: (N,3,H,W) in [-1,1] -> (N,49,768) CLIP patch tokens."""
    outs = []
    for s in range(0, len(images), batch):
        xb = images[s:s + batch].to(DEVICE)
        xb = (xb + 1) / 2
        xb = F.interpolate(xb, size=224, mode="bilinear", align_corners=False)
        xb = (xb - CLIP_MEAN) / CLIP_STD
        h = clip(pixel_values=xb).last_hidden_state    # (B, 50, 768)
        outs.append(h[:, 1:, :].cpu())                 # drop CLS
    return torch.cat(outs)


def load_folder(folder, n, size):
    from PIL import Image
    paths = sorted(glob.glob(os.path.join(folder, "*")))[:n]
    ims = []
    for p in paths:
        im = Image.open(p).convert("RGB").resize((size, size))
        ims.append(torch.tensor(np.array(im)).permute(2, 0, 1))
    return torch.stack(ims).float() / 127.5 - 1.0


# ---------------------------------------------------------------- reference images
if os.path.exists(X0_PATH):
    x0 = torch.load(X0_PATH)
    print(f"Loaded cached reference images: {x0.shape}")
elif IMAGE_DIR is not None:
    x0 = load_folder(IMAGE_DIR, N_IMAGES, IMG_SIZE)
    torch.save(x0, X0_PATH)
else:
    print("Sampling reference images from the DDPM prior ...")
    gen = []
    with torch.no_grad():
        for s in range(0, N_IMAGES, BATCH):
            b = min(BATCH, N_IMAGES - s)
            x = torch.randn(b, 3, IMG_SIZE, IMG_SIZE, device=DEVICE)
            scheduler.set_timesteps(T_TOTAL)
            for t in scheduler.timesteps:
                eps = unet(x, t).sample
                x = scheduler.step(eps, t, x).prev_sample
            gen.append(x.cpu())
            print(f"  generated {s + b}/{N_IMAGES}")
    x0 = torch.cat(gen)[:N_IMAGES]
    torch.save(x0, X0_PATH)   # cache so resumed jobs reuse the SAME x0
print(f"x0: {tuple(x0.shape)}")

emb0 = patch_embeddings(x0)
print(f"Original patch embeddings: {tuple(emb0.shape)}")

# ---------------------------------------------------------------- main loop
if os.path.exists(SAVE_PATH):
    ck = np.load(SAVE_PATH)
    delta_norms = ck["delta_norms"]
    done = ck["done"]
    print(f"Resuming: {done.sum()}/{done.size} runs done")
else:
    delta_norms = np.zeros((len(T_STARS), N_RUNS, N_IMAGES, N_PATCH))
    done = np.zeros((len(T_STARS), N_RUNS), dtype=bool)

for i, t_star in enumerate(T_STARS):
    for r in range(N_RUNS):
        if done[i, r]:
            continue
        print(f"t*={t_star} ({T_FRACS[i]:.2f}T)  run {r + 1}/{N_RUNS} ...", flush=True)
        xhat = forward_backward(x0, t_star)
        emb_hat = patch_embeddings(xhat)
        delta_norms[i, r] = (emb0 - emb_hat).norm(dim=-1).numpy()
        done[i, r] = True
        np.savez(SAVE_PATH, delta_norms=delta_norms, done=done,
                 t_stars=np.array(T_STARS), t_fracs=np.array(T_FRACS))
        print(f"  saved  [{done.sum()}/{done.size}]", flush=True)

print("All runs complete.")

# ---------------------------------------------------------------- analysis
import matplotlib                                            # noqa: E402
matplotlib.use("Agg")                                        # headless cluster
import matplotlib.pyplot as plt                               # noqa: E402
from scipy.optimize import curve_fit                          # noqa: E402

n_T = len(T_STARS)
C_all = np.zeros((n_T, N_PATCH, N_PATCH))
chi = np.zeros(n_T)

for i in range(n_T):
    dn = delta_norms[i].reshape(-1, N_PATCH)
    dn = dn[~np.isnan(dn).any(axis=1)]
    mean_i = dn.mean(axis=0)
    C = (dn[:, :, None] * dn[:, None, :]).mean(axis=0) - np.outer(mean_i, mean_i)
    C_all[i] = C
    chi[i] = C.sum() / np.trace(C)

np.savez(f"{OUT_PREFIX}_correlators.npz", C_all=C_all, chi=chi,
         t_fracs=np.array(T_FRACS))

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(T_FRACS, chi, marker="o")
ax.set(xlabel="t* / T", ylabel=r"$\chi$", title="Susceptibility vs inversion time")
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT_PREFIX}_chi_vs_t.png", dpi=150)

peak = int(np.argmax(chi))
print(f"chi peaks at t*/T = {T_FRACS[peak]}  (chi = {chi[peak]:.3f})")

# spatial correlation function + correlation length
coords = np.array([(k // GRID, k % GRID) for k in range(N_PATCH)])
dist_matrix = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
bins = np.arange(0, dist_matrix.max() + 1.0, 1.0)
bin_idx = np.digitize(dist_matrix, bins) - 1
r_axis = 0.5 * (bins[:-1] + bins[1:])

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
xi_vals = []
for i in range(n_T):
    C = C_all[i]
    d = np.sqrt(np.diag(C))
    rho = C / np.outer(d, d)
    corr_vs_r = np.array([rho[bin_idx == b].mean() for b in range(len(bins) - 1)])
    axes[0].plot(r_axis, corr_vs_r, marker="o", markersize=3, alpha=0.7,
                 label=f"t*/T={T_FRACS[i]}")
    try:
        popt, _ = curve_fit(lambda r, xi, A: A * np.exp(-r / xi),
                            r_axis[1:], corr_vs_r[1:],
                            p0=[1.0, max(corr_vs_r[1], 1e-3)], maxfev=5000)
        xi_vals.append(popt[0])
    except Exception:
        xi_vals.append(np.nan)

axes[0].set(xlabel="patch distance r", ylabel="normalized correlation",
            title="Spatial correlation function")
axes[0].legend(fontsize=6)
axes[1].plot(T_FRACS, xi_vals, marker="o")
axes[1].set(xlabel="t* / T", ylabel="correlation length (patches)",
            title="Correlation length vs inversion time")
axes[1].grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT_PREFIX}_correlation_length.png", dpi=150)

np.savez(f"{OUT_PREFIX}_xi.npz", xi=np.array(xi_vals), t_fracs=np.array(T_FRACS))
print("Analysis complete. Outputs:",
      f"{OUT_PREFIX}_chi_vs_t.png, {OUT_PREFIX}_correlation_length.png,",
      f"{OUT_PREFIX}_correlators.npz, {OUT_PREFIX}_xi.npz")
