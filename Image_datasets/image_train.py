"""
Train a pixel-space DDPM on a pre-made image tensor (e.g. CelebA, LSUN).

Expects a tensor file of shape (N, 3, H, W) in [-1, 1], e.g. produced by
your make_dataset.py that downloads via HF datasets and saves a .pt.

Usage:
  # compute node (SLURM) -- no internet needed, reads the local .pt:
  python3 train_ddpm_tensor.py --data celeba_20k_32px.pt --tag celeba

  # two-class experiment: concatenate two tensors first (see note at bottom),
  # or pass a combined tensor and use --tag celeba_church

Output: ddpm_<tag>/{unet, scheduler, x0_holdout.pt, train_state.pt}
Checkpointed per epoch -- resubmit after timeout to resume.
Stops early once ~1M images have been seen (configurable).
"""

import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# ---------------------------------------------------------------- args
parser = argparse.ArgumentParser()
parser.add_argument("--data", required=True, help="path to (N,3,H,W) tensor in [-1,1]")
parser.add_argument("--tag", required=True, help="output name: ddpm_<tag>/")
parser.add_argument("--epochs", type=int, default=200)
parser.add_argument("--batch", type=int, default=512)
parser.add_argument("--lr", type=float, default=2e-4)
parser.add_argument("--max-images", type=int, default=10_000_000,
                    help="stop once this many image-presentations reached")
parser.add_argument("--n-holdout", type=int, default=1000,
                    help="images held out as x0 references (never trained on)")
args = parser.parse_args()

OUT_DIR = f"ddpm_{args.tag}"
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------- data
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", DEVICE)

images = torch.load(args.data, map_location="cpu").float()
assert images.dim() == 4 and images.shape[1] == 3, \
    f"expected (N,3,H,W), got {tuple(images.shape)}"
IMG_SIZE = images.shape[-1]
print(f"Loaded {len(images)} images, shape {tuple(images.shape)}, "
      f"range [{images.min():.2f}, {images.max():.2f}]")

# held-out references (deterministic split)
perm = torch.randperm(len(images), generator=torch.Generator().manual_seed(0))
images = images[perm]
x0_holdout, train_imgs = images[:args.n_holdout], images[args.n_holdout:]
torch.save(x0_holdout, os.path.join(OUT_DIR, "x0_holdout.pt"))
print(f"Held out {len(x0_holdout)} reference images -> {OUT_DIR}/x0_holdout.pt")

#loader = DataLoader(TensorDataset(train_imgs), batch_size=args.batch,
#                    shuffle=True, num_workers=0, pin_memory=True, persistent_workers=True, drop_last=True)


# GPU-resident dataset — no DataLoader needed
train_imgs = train_imgs.to(DEVICE)
n_train = len(train_imgs)
print(f"Dataset moved to GPU: {train_imgs.shape}, "
      f"{train_imgs.element_size() * train_imgs.nelement() / 1e9:.1f} GB")

# manual batching, no DataLoader:
#perm = torch.randperm(len(train_imgs), device=DEVICE)
#for start in range(0, len(train_imgs), args.batch):
#    xb = train_imgs[perm[start:start+args.batch]]

    # ... training step ...
# ---------------------------------------------------------------- model
from diffusers import UNet2DModel, DDPMScheduler

unet = UNet2DModel(
    sample_size=IMG_SIZE, in_channels=3, out_channels=3,
    layers_per_block=2,
    block_out_channels=(128, 256, 256, 256),
    down_block_types=("DownBlock2D", "AttnDownBlock2D",
                      "DownBlock2D", "DownBlock2D"),
    up_block_types=("UpBlock2D", "UpBlock2D",
                    "AttnUpBlock2D", "UpBlock2D"),
).to(DEVICE)

scheduler = DDPMScheduler(num_train_timesteps=1000)
opt = torch.optim.AdamW(unet.parameters(), lr=args.lr)

# resume
start_epoch, images_seen = 0, 0
ckpt_path = os.path.join(OUT_DIR, "train_state.pt")
if os.path.exists(ckpt_path):
    st = torch.load(ckpt_path, map_location=DEVICE)
    unet.load_state_dict(st["unet"])
    opt.load_state_dict(st["opt"])
    start_epoch = st["epoch"] + 1
    images_seen = st.get("images_seen", 0)
    print(f"Resuming from epoch {start_epoch}, images_seen={images_seen}")

# ---------------------------------------------------------------- training
#T = scheduler.config.num_train_timesteps
#for epoch in range(start_epoch, args.epochs):
#    losses = []
#    for (xb,) in loader:
#        xb = xb.to(DEVICE)
#        t = torch.randint(0, T, (xb.shape[0],), device=DEVICE)
#        noise = torch.randn_like(xb)
#        xt = scheduler.add_noise(xb, noise, t)
#        pred = unet(xt, t).sample
#        loss = F.mse_loss(pred, noise)
#        opt.zero_grad(); loss.backward(); opt.step()
#        losses.append(loss.item())
#        images_seen += xb.shape[0]

import time
T = scheduler.config.num_train_timesteps
for epoch in range(start_epoch, args.epochs):
    losses = []
    perm = torch.randperm(n_train, device=DEVICE)   # fresh shuffle each epoch
#    epoch_start=time.time()

    for step,start in enumerate(range(0, n_train - args.batch + 1, args.batch)):
#        t0=time.time()

        xb = train_imgs[perm[start:start + args.batch]]
        t = torch.randint(0, T, (xb.shape[0],), device=DEVICE)
        noise = torch.randn_like(xb)
        xt = scheduler.add_noise(xb, noise, t)
#        torch.cuda.synchronize()
#        t_data = time.time() - t0

        pred = unet(xt, t).sample
        loss = F.mse_loss(pred, noise)
#        torch.cuda.synchronize()
#        t_fwd = time.time() - t0 - t_data

        opt.zero_grad(); loss.backward()
#        torch.cuda.synchronize()
#        t_bwd = time.time() - t0 - t_data - t_fwd

        opt.step()
 #       torch.cuda.synchronize()
 #       t_opt = time.time() - t0 - t_data - t_fwd - t_bwd

        losses.append(loss.item())
        images_seen += xb.shape[0]

  #      if step % 50 == 0:
  #          print(f"  step {step:3d}  data={t_data*1000:5.0f}ms  "
  #                f"fwd={t_fwd*1000:5.0f}ms  bwd={t_bwd*1000:5.0f}ms  "
  #                f"opt={t_opt*1000:4.0f}ms  total={(time.time()-t0)*1000:5.0f}ms",
  #                flush=True)


    print(f"epoch {epoch+1}  loss={np.mean(losses):.5f}  "
          f"images_seen={images_seen}", flush=True)
#    print(f"epoch {epoch+1}  loss={np.mean(losses):.5f}  "
#              f"time={epoch_time:.1f}s  "
#              f"peak_gpu={torch.cuda.max_memory_allocated()/1e9:.1f}GB  "
#              f"images_seen={images_seen}", flush=True)
#    torch.cuda.reset_peak_memory_stats()

    torch.save({"unet": unet.state_dict(), "opt": opt.state_dict(),
                "epoch": epoch, "images_seen": images_seen}, ckpt_path)

    if (epoch + 1) % 10 == 0:
        unet.save_pretrained(os.path.join(OUT_DIR, "unet"))
        scheduler.save_pretrained(os.path.join(OUT_DIR, "scheduler"))
        print(f"  snapshot saved -> {OUT_DIR}/unet", flush=True)

    if images_seen >= args.max_images:
        print(f"Reached {images_seen} images seen -- stopping early.")
        break

unet.save_pretrained(os.path.join(OUT_DIR, "unet"))
scheduler.save_pretrained(os.path.join(OUT_DIR, "scheduler"))
print(f"Done. Model in {OUT_DIR}/unet")

# ------------------------------------------------------------------------
# TWO-CLASS NOTE: to build a combined dataset, on the login node do e.g.
#   a = torch.load('celeba_20k_32px.pt')[:20000]
#   b = torch.load('church_20k_32px.pt')[:20000]   # equal counts!
#   torch.save(torch.cat([a, b]), 'celeba_church_40k.pt')
# then: python3 train_ddpm_tensor.py --data celeba_church_40k.pt --tag celeba_church
