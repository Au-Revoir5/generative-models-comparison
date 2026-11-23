"""Generate Figures 1, 2, 4, 5 (Figure 3 comes from experiments/mlp_limitations.py)
plus an extra Sparse-AE lambda-sweep plot. No training: uses saved checkpoints/results.

Needs: all checkpoints, results/*.json, results/final_metrics.json,
       checkpoints/ddpm_samples.pt and outputs/ddpm_trajectory.png
Run:   python -m figures.make_figures
"""
import json, os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.utils import make_grid

from data import CLASSES, get_loaders, to_01
from models.ae import VanillaAE
from models.dcgan import Generator as DCGenerator
from models.ddpm import GaussianDiffusion, UNet
from models.gan import Generator as MLPGenerator
from models.mlp import MLPClassifier
from models.sparse_ae import SparseAE
from models.vae import VAE
from utils import ensure_dir, get_device, set_seed

OUT = "outputs"
SEED = 42
ALPHAS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
CLASS_A, CLASS_B = 7, 0          # Sneaker -> T-shirt/top (visually contrasting)


# ------------------------------------------------------------------ helpers
def rj(path):
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def load(model, path, device):
    model.load_state_dict(torch.load(path, map_location=device))
    return model.to(device).eval()


def show(ax, imgs, nrow, title=None, title_loc="left", fontsize=10):
    grid = make_grid(imgs.detach().float().cpu().clamp(0, 1), nrow=nrow, padding=2, pad_value=1.0)
    ax.imshow(grid.permute(1, 2, 0).numpy())
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=fontsize, loc=title_loc)


def smooth(y, w):
    y = np.asarray(y, dtype=float)
    if len(y) < w:
        return y
    return np.convolve(y, np.ones(w) / w, mode="valid")


@torch.no_grad()
def one_per_class(loader, n=8):
    chosen = {}
    for x, y in loader:
        for img, lab in zip(x, y):
            c = lab.item()
            if c < n and c not in chosen:
                chosen[c] = img
            if len(chosen) == n:
                return torch.stack([chosen[c] for c in range(n)])


@torch.no_grad()
def first_of_class(loader, c):
    for x, y in loader:
        idx = (y == c).nonzero()
        if len(idx):
            return x[idx[0, 0]:idx[0, 0] + 1]


@torch.no_grad()
def class_mean(loader, c):
    xs = [x[y == c] for x, y in loader]
    return torch.cat(xs).mean(0, keepdim=True)


@torch.no_grad()
def pick_prior_z(G, clf, z_dim, cls, device, mean_img, n=2000):
    """No encoder in a GAN: among prior samples the classifier assigns to `cls`
    with p > 0.9, pick the one closest (pixel MSE) to the real class-mean image.
    (Taking the single most confident sample tends to select odd hybrid images
    that merely fool the classifier.)"""
    z = torch.randn(n, z_dim, device=device)
    imgs = to_01(G(z))
    prob = F.softmax(clf(imgs), 1)[:, cls]
    dist = ((imgs - mean_img.to(device)) ** 2).flatten(1).mean(1)
    dist[prob < 0.9] = float("inf")
    idx = dist.argmin() if torch.isfinite(dist.min()) else prob.argmax()
    return z[idx].unsqueeze(0)


def lerp(za, zb):
    return torch.cat([(1 - a) * za + a * zb for a in ALPHAS])


# ------------------------------------------------------------------ figures
@torch.no_grad()
def figure1(ae, sae, vae, test_loader, device):
    R = {k: rj(f"results/{k}.json").get("test", {}) for k in ("vanilla_ae", "sparse_ae", "vae")}
    x = one_per_class(test_loader).to(device)
    rows = [
        (x, "Ground truth (test set): " + ", ".join(CLASSES[:8])),
        (ae(x), "Vanilla AE   (test MSE {mse:.5f} | PSNR {psnr:.2f} dB | SSIM {ssim:.4f})".format(**R["vanilla_ae"])),
        (sae(x)[0], "Sparse AE, λ=1e-3   (test MSE {mse:.5f} | PSNR {psnr:.2f} dB | SSIM {ssim:.4f})".format(**R["sparse_ae"])),
        (vae.reconstruct(x), "VAE, z = μ   (test MSE {mse:.5f} | PSNR {psnr:.2f} dB | SSIM {ssim:.4f})".format(**R["vae"])),
    ]
    fig, axes = plt.subplots(len(rows), 1, figsize=(11, 1.65 * len(rows) + 0.5))
    for ax, (imgs, t) in zip(axes, rows):
        show(ax, imgs, nrow=8, title=t)
    fig.suptitle("Figure 1. Side-by-side reconstructions on the test set", fontsize=12, y=1.0)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig1_reconstruction.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


@torch.no_grad()
def figure2(vae, gan, dcgan, device):
    fid = rj("results/final_metrics.json").get("fid", {})
    torch.manual_seed(SEED)
    panels = [
        ("A", "VAE", vae.sample(16, device)),
        ("B", "Vanilla GAN", to_01(gan(torch.randn(16, 100, device=device)))),
        ("C", "DCGAN", to_01(dcgan(torch.randn(16, 100, device=device)))),
        ("D", "DDPM", torch.load("checkpoints/ddpm_samples.pt")[:16]),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.6))
    for ax, (letter, name, imgs) in zip(axes, panels):
        f = fid.get(name)
        sub = f"FID {f:.2f}" if f is not None else ""
        src = "x_T ~ N(0, I), 1000 reverse steps" if name == "DDPM" else "z ~ N(0, I)"
        show(ax, imgs, nrow=4, title=f"Panel {letter}: {name}\n{src}   {sub}", title_loc="center")
    fig.suptitle("Figure 2. Random generative sampling (4x4 per model)", fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig2_samples.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


@torch.no_grad()
def figure4(ae, sae, vae, gan, dcgan, clf, test_loader, device):
    xa = first_of_class(test_loader, CLASS_A).to(device)
    xb = first_of_class(test_loader, CLASS_B).to(device)
    ma, mb = class_mean(test_loader, CLASS_A), class_mean(test_loader, CLASS_B)
    torch.manual_seed(SEED)
    rows = [
        ("Vanilla AE", ae.decode(lerp(ae.encode(xa), ae.encode(xb)))),
        ("Sparse AE", sae.decode(lerp(sae.encode(xa), sae.encode(xb)))),
        ("VAE (μ)", vae.decode(lerp(vae.encode(xa)[0], vae.encode(xb)[0]))),
        ("Vanilla GAN", to_01(gan(lerp(pick_prior_z(gan, clf, 100, CLASS_A, device, ma),
                                       pick_prior_z(gan, clf, 100, CLASS_B, device, mb))))),
        ("DCGAN", to_01(dcgan(lerp(pick_prior_z(dcgan, clf, 100, CLASS_A, device, ma),
                                   pick_prior_z(dcgan, clf, 100, CLASS_B, device, mb))))),
    ]
    # Confidence of the classifier on each frame: dips at the midpoint suggest the
    # path left the data manifold (latent voids).
    fig, axes = plt.subplots(len(rows), 1, figsize=(8, 1.75 * len(rows) + 0.9))
    conf_report = {}
    for ax, (name, imgs) in zip(axes, rows):
        prob = F.softmax(clf(imgs), 1)
        conf = prob.max(1)[0].cpu().numpy()
        conf_report[name] = [round(float(c), 3) for c in conf]
        show(ax, imgs, nrow=len(ALPHAS),
             title=f"{name}   (classifier confidence per frame: " + ", ".join(f"{c:.2f}" for c in conf) + ")",
             fontsize=9)
    fig.suptitle(f"Figure 4. Linear latent interpolation z(α) = (1-α) z_A + α z_B,  "
                 f"α = {ALPHAS}\n{CLASSES[CLASS_A]} → {CLASSES[CLASS_B]} "
                 "(AE/VAE: encoded test images; GANs: typical prior samples of each class, see code)",
                 fontsize=10, y=1.02)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig4_interpolation.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    with open("results/interpolation_confidence.json", "w") as f:
        json.dump(conf_report, f, indent=2)


@torch.no_grad()
def ddpm_snapshots(device, n=8, snap_ts=(800, 600, 400, 200, 100, 50, 0)):
    if not os.path.exists("checkpoints/ddpm_ema.pt"):
        return None
    model = load(UNet(), "checkpoints/ddpm_ema.pt", device)
    diffusion = GaussianDiffusion(T=1000, device=device)
    torch.manual_seed(SEED)
    _, snaps = diffusion.sample(model, n, device=device, snapshot_ts=snap_ts)
    return {t: to_01(x).cpu() for t, x in snaps.items()}


def figure5(snaps=None):
    ae, sae, vae = rj("results/vanilla_ae.json"), rj("results/sparse_ae.json"), rj("results/vae.json")
    gan, dcgan, ddpm = rj("results/gan.json"), rj("results/dcgan.json"), rj("results/ddpm.json")

    fig = plt.figure(figsize=(17, 15))
    gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 1.2], hspace=0.38, wspace=0.42,
                          top=0.94, bottom=0.03, left=0.05, right=0.96)

    # (a) Vanilla AE
    ax = fig.add_subplot(gs[0, 0]); h = ae["history"]; ep = np.arange(1, len(h["train_loss"]) + 1)
    ax.plot(ep, h["train_loss"], label="train MSE"); ax.plot(ep, h["val_loss"], label="val MSE")
    ax.set_title("(a) Vanilla AE"); ax.set_xlabel("epoch"); ax.set_ylabel("MSE"); ax.legend()

    # (b) Sparse AE
    ax = fig.add_subplot(gs[0, 1]); h = sae["history"]; ep = np.arange(1, len(h["train_loss"]) + 1)
    ax.plot(ep, h["train_mse"], label="train MSE"); ax.plot(ep, h["val_mse"], label="val MSE")
    ax.plot(ep, h["train_loss"], "--", label="train MSE + λ·L1")
    ax2 = ax.twinx(); ax2.plot(ep, h["train_l1"], color="gray", alpha=0.6, label="mean |h| (right)")
    ax2.set_ylabel("mean |h|")
    ax.set_title("(b) Sparse AE (λ = 1e-3)"); ax.set_xlabel("epoch"); ax.set_ylabel("loss")
    l1, n1 = ax.get_legend_handles_labels(); l2, n2 = ax2.get_legend_handles_labels()
    ax.legend(l1 + l2, n1 + n2, fontsize=8)

    # (c) VAE
    ax = fig.add_subplot(gs[0, 2]); h = vae["history"]; ep = np.arange(1, len(h["train_loss"]) + 1)
    ax.plot(ep, h["train_recon"], label="train BCE (recon)"); ax.plot(ep, h["val_loss"], label="val total")
    ax.plot(ep, h["train_loss"], "--", label="train total")
    ax2 = ax.twinx(); ax2.plot(ep, h["train_kl"], color="gray", alpha=0.7, label="train KL (right)")
    ax2.set_ylabel("KL (nats)")
    ax.set_title("(c) VAE"); ax.set_xlabel("epoch"); ax.set_ylabel("nats per image")
    l1, n1 = ax.get_legend_handles_labels(); l2, n2 = ax2.get_legend_handles_labels()
    ax.legend(l1 + l2, n1 + n2, fontsize=8)

    # (d), (e) GANs: per-iteration, raw (faint) + moving average
    for k, (res, name) in enumerate(((gan, "(d) Vanilla GAN (MLP)"), (dcgan, "(e) DCGAN"))):
        ax = fig.add_subplot(gs[1, k]); it = res["iter_history"]
        n_ep = len(res["epoch_history"]["loss_D"]); x = np.arange(len(it["loss_D"])) / (len(it["loss_D"]) / n_ep)
        for key, lab in (("loss_D", "D loss"), ("loss_G", "G loss")):
            line, = ax.plot(x, it[key], alpha=0.15, lw=0.5)
            s = smooth(it[key], 200)
            ax.plot(x[len(x) - len(s):], s, color=line.get_color(), label=f"{lab} (moving avg)")
        ax.axhline(2 * np.log(2), ls=":", color="k", lw=1, label="2·ln2 (D at equilibrium)")
        ax.axhline(np.log(2), ls="--", color="k", lw=0.8, label="ln2 (G at equilibrium)")
        ax.set_title(name); ax.set_xlabel("epoch"); ax.set_ylabel("BCE loss"); ax.legend(fontsize=8)

    # (f) DDPM
    ax = fig.add_subplot(gs[1, 2]); h = ddpm["history"]; ep = np.arange(1, len(h["train_loss"]) + 1)
    it = ddpm["iter_loss"]; x = np.arange(len(it)) / (len(it) / len(ep))
    ax.plot(x, it, alpha=0.12, lw=0.5, color="C0")
    s = smooth(it, 200); ax.plot(x[len(x) - len(s):], s, color="C0", label="train noise MSE (moving avg)")
    ax.plot(ep, h["val_loss"], "o-", ms=3, color="C1", label="val noise MSE (EMA)")
    ax.set_yscale("log"); ax.set_title("(f) DDPM  ||ε − ε_θ(x_t, t)||²")
    ax.set_xlabel("epoch"); ax.set_ylabel("MSE (log scale)"); ax.legend(fontsize=8)

    # (g) DDPM reverse trajectory: one row per timestep snapshot, columns = samples
    if snaps:
        ts = sorted(snaps.keys(), reverse=True)
        sub = gs[2, :].subgridspec(1, len(ts), wspace=0.04)
        for j, t in enumerate(ts):
            ax = fig.add_subplot(sub[0, j])
            show(ax, snaps[t], nrow=2, title=("x_T (t=1000)" if t == max(ts) else
                                              ("x_0 (t=0)" if t == 0 else f"t = {t}")),
                 title_loc="center", fontsize=11)
        top = max(a.get_position().y1 for a in fig.axes[-len(ts):])
        fig.text(0.5, top + 0.03, "(g) DDPM reverse diffusion x_T → x_0 (EMA model, 8 samples per column; "
                 "coarse shape appears first, fine detail last)", ha="center", fontsize=12)

    fig.suptitle("Figure 5. Training loss curves and DDPM denoising trajectory", fontsize=14, y=0.985)
    fig.savefig(f"{OUT}/fig5_training_curves.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def extra_sparse_sweep():
    runs = [("0\n(Vanilla AE)", rj("results/vanilla_ae.json").get("test"), None)]
    base = rj("results/sparse_ae.json")
    if base:
        runs[0] = (runs[0][0], runs[0][1], base.get("vanilla_ae_hidden_sparsity", {}).get("avg_active_units_per_image"))
    for lab, name in (("1e-3", "sparse_ae"), ("1e-2", "sparse_ae_lam1e-2"), ("1e-1", "sparse_ae_lam1e-1")):
        r = rj(f"results/{name}.json")
        if r:
            runs.append((lab, r["test"], r["sparsity"]["avg_active_units_per_image"]))
    if len(runs) < 2:
        return
    labels = [r[0] for r in runs]
    x = np.arange(len(runs))
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(x, [r[1]["mse"] for r in runs], "o-", color="C0", label="test MSE")
    ax.set_ylabel("test MSE", color="C0"); ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_xlabel("λ")
    ax2 = ax.twinx()
    ax2.plot(x, [r[2] for r in runs], "s--", color="C3", label="active hidden units / 256")
    ax2.set_ylabel("active hidden units per image (of 256)", color="C3")
    for i, r in enumerate(runs):
        ax.annotate(f"SSIM {r[1]['ssim']:.3f}", (x[i], r[1]["mse"]), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=8)
    ax.set_title("Sparse AE: sparsity vs reconstruction fidelity (λ sweep)")
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_extra_sparse_sweep.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    set_seed(SEED)
    device = get_device()
    ensure_dir(OUT)
    _, _, test_loader = get_loaders("01", 128)

    ae = load(VanillaAE(64), "checkpoints/vanilla_ae.pt", device)
    sae = load(SparseAE(64), "checkpoints/sparse_ae.pt", device)
    vae = load(VAE(64), "checkpoints/vae.pt", device)
    gan = load(MLPGenerator(100), "checkpoints/gan_G.pt", device)
    dcgan = load(DCGenerator(100), "checkpoints/dcgan_G.pt", device)
    clf = load(MLPClassifier(), "checkpoints/mlp_classifier.pt", device)

    figure1(ae, sae, vae, test_loader, device);                print("Saved Figure 1")
    figure2(vae, gan, dcgan, device);                         print("Saved Figure 2")
    figure4(ae, sae, vae, gan, dcgan, clf, test_loader, device); print("Saved Figure 4")
    print("Sampling DDPM trajectory for Figure 5 (~45 s)...")
    figure5(ddpm_snapshots(device));                          print("Saved Figure 5")
    extra_sparse_sweep();                                     print("Saved extra sparse sweep plot")


if __name__ == "__main__":
    main()
