"""Section 2, Stages 2-5: empirical proof that MLP-classifier weights and latent
features cannot be used directly for image reconstruction or generation.

Requires (run first):
    python -m train.train_mlp
    python -m train.train_ae        (for the z_AE comparison in Stage 4)

Run from the project root:  python -m experiments.mlp_limitations

Outputs:
    outputs/fig3_mlp_limitations.png
    results/mlp_limitations.json
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torchvision.utils import make_grid

from data import CLASSES, get_loaders
from evaluation.metrics import reconstruction_metrics
from models.ae import VanillaAE, make_decoder
from models.mlp import MLPClassifier
from utils import ensure_dir, get_device, save_json, set_seed

MLP_CKPT = "checkpoints/mlp_classifier.pt"
AE_CKPT = "checkpoints/vanilla_ae.pt"
DEC_CKPT = "checkpoints/mlp_frozen_decoder.pt"


# ---------------------------------------------------------------- Stage 2 & 3
def transpose_from_class(W1, W2, W3):
    """x_hat = Sigmoid(W1^T ReLU(W2^T ReLU(W3^T y_c))) for every one-hot y_c."""
    Y = torch.eye(W3.shape[0], device=W3.device)        # (10, 10) one-hot rows
    z_hat = F.relu(Y @ W3)                              # (10, 64)
    h1_hat = F.relu(z_hat @ W2)                         # (10, 256)
    return torch.sigmoid(h1_hat @ W1).view(-1, 1, 28, 28)


def transpose_from_latent(z, W1, W2):
    """x_hat = Sigmoid(W1^T ReLU(W2^T z_MLP))."""
    return torch.sigmoid(F.relu(z @ W2) @ W1).view(-1, 1, 28, 28)


def pinv_from_latent(z, W1, W2, b1, b2):
    """Extra (not required): best-effort *linear* inverse with the Moore-Penrose
    pseudo-inverse. Still fails, because ReLU destroyed the negative parts and
    rank(W1) <= 256 < 784."""
    h1 = (z - b2) @ torch.linalg.pinv(W2).T
    x = (h1 - b1) @ torch.linalg.pinv(W1).T
    return x.view(-1, 1, 28, 28)


def orthogonality_report(W):
    """How far W^T W is from the identity, plus rank / conditioning."""
    WtW = W.T @ W
    I = torch.eye(WtW.shape[0], device=W.device)
    s = torch.linalg.svdvals(W)
    return {
        "shape_out_in": list(W.shape),
        "rel_frobenius_error_WtW_vs_I": (torch.linalg.norm(WtW - I) / torch.linalg.norm(I)).item(),
        "rank": int(torch.linalg.matrix_rank(W).item()),
        "max_possible_rank": min(W.shape),
        "input_dim": W.shape[1],
        "sigma_max": s.max().item(),
        "sigma_min": s.min().item(),
        "condition_number": (s.max() / s.min()).item(),
    }


@torch.no_grad()
def relu_zero_fraction(mlp, loader, device):
    """Fraction of activations ReLU set to exactly zero (information lost irreversibly)."""
    zeros_h1 = zeros_z = total_h1 = total_z = 0
    for x, _ in loader:
        x = x.to(device)
        h1 = mlp.hidden1(x)
        z = mlp.features(x)
        zeros_h1 += (h1 == 0).sum().item(); total_h1 += h1.numel()
        zeros_z += (z == 0).sum().item(); total_z += z.numel()
    return {"h1_zero_fraction": zeros_h1 / total_h1, "z_zero_fraction": zeros_z / total_z}


# ---------------------------------------------------------------- Stage 4
def train_frozen_decoder(mlp, train_loader, val_loader, device, epochs, lr):
    for p in mlp.parameters():
        p.requires_grad_(False)
    mlp.eval()

    decoder = make_decoder(latent_dim=64, hidden=256).to(device)   # same as AE decoder
    opt = torch.optim.Adam(decoder.parameters(), lr=lr)
    best = float("inf")
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(1, epochs + 1):
        decoder.train()
        total, n = 0.0, 0
        for x, _ in train_loader:
            x = x.to(device)
            with torch.no_grad():
                z = mlp.features(x)
            loss = F.mse_loss(decoder(z), x)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * x.size(0)
            n += x.size(0)

        decoder.eval()
        with torch.no_grad():
            v_total, v_n = 0.0, 0
            for x, _ in val_loader:
                x = x.to(device)
                v_total += F.mse_loss(decoder(mlp.features(x)), x).item() * x.size(0)
                v_n += x.size(0)
        val = v_total / v_n
        history["train_loss"].append(total / n)
        history["val_loss"].append(val)
        print(f"[frozen decoder] epoch {epoch:02d}  train_mse {total / n:.5f}  val_mse {val:.5f}")
        if val < best:
            best = val
            torch.save(decoder.state_dict(), DEC_CKPT)

    decoder.load_state_dict(torch.load(DEC_CKPT, map_location=device))
    decoder.eval()
    return decoder, history


# ---------------------------------------------------------------- helpers
@torch.no_grad()
def pick_one_per_class(loader, n=8):
    """First test image of each of the first n classes (deterministic)."""
    chosen = {}
    for x, y in loader:
        for img, label in zip(x, y):
            c = label.item()
            if c < n and c not in chosen:
                chosen[c] = img
            if len(chosen) == n:
                return torch.stack([chosen[c] for c in range(n)]), list(range(n))
    raise RuntimeError("Not enough classes found")


def minmax_per_image(x):
    flat = x.flatten(1)
    lo, hi = flat.min(1, keepdim=True)[0], flat.max(1, keepdim=True)[0]
    return ((flat - lo) / (hi - lo + 1e-8)).view_as(x)


def show_row(ax, imgs, title, nrow=None):
    grid = make_grid(imgs.detach().cpu().clamp(0, 1), nrow=nrow or imgs.size(0),
                     padding=2, pad_value=1.0)
    ax.imshow(grid.permute(1, 2, 0).numpy(), cmap="gray", vmin=0, vmax=1)
    ax.set_title(title, fontsize=10, loc="left")
    ax.axis("off")


# ---------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--decoder-epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    set_seed(args.seed)
    device = get_device()
    ensure_dir("outputs")
    train_loader, val_loader, test_loader = get_loaders("01", 128)

    mlp = MLPClassifier().to(device)
    mlp.load_state_dict(torch.load(MLP_CKPT, map_location=device))
    mlp.eval()
    W1, W2, W3 = mlp.weights()
    b1, b2, _ = mlp.biases()
    results = {}

    # Theory checks on the weights (supports the W^T != W^-1 argument)
    results["orthogonality"] = {name: orthogonality_report(W)
                                for name, W in (("W1", W1), ("W2", W2), ("W3", W3))}
    results["relu_information_loss"] = relu_zero_fraction(mlp, test_loader, device)

    x_gt, labels = pick_one_per_class(test_loader, 8)
    x_gt = x_gt.to(device)

    with torch.no_grad():
        # Stage 2: top-down weight transposition from class vectors
        x_class = transpose_from_class(W1, W2, W3)
        results["stage2_class_projection"] = {
            "pixel_mean": x_class.mean().item(),
            "pixel_std": x_class.std().item(),
            "per_class_std_across_classes": x_class.flatten(1).std(0).mean().item(),
        }

        # Stage 3: latent transpose projection from real test images
        z_gt = mlp.features(x_gt)
        x_lat_T = transpose_from_latent(z_gt, W1, W2)
        x_lat_pinv = pinv_from_latent(z_gt, W1, W2, b1, b2)

    results["stage3_transpose_test_metrics"] = reconstruction_metrics(
        lambda x: transpose_from_latent(mlp.features(x), W1, W2), test_loader, device)
    results["extra_pinv_test_metrics"] = reconstruction_metrics(
        lambda x: pinv_from_latent(mlp.features(x), W1, W2, b1, b2), test_loader, device)

    # Stage 4: frozen latent decoding vs Vanilla AE with the same latent size
    decoder, dec_hist = train_frozen_decoder(mlp, train_loader, val_loader, device,
                                             args.decoder_epochs, args.lr)
    results["stage4_frozen_decoder_test_metrics"] = reconstruction_metrics(
        lambda x: decoder(mlp.features(x)), test_loader, device)
    results["stage4_frozen_decoder_history"] = dec_hist

    ae = None
    if os.path.exists(AE_CKPT):
        ae = VanillaAE(latent_dim=64).to(device)
        ae.load_state_dict(torch.load(AE_CKPT, map_location=device))
        ae.eval()
        results["stage4_vanilla_ae_test_metrics"] = reconstruction_metrics(ae, test_loader, device)
    else:
        print(f"WARNING: {AE_CKPT} not found - run `python -m train.train_ae` for the AE comparison.")

    with torch.no_grad():
        x_frozen = decoder(z_gt)
        x_ae = ae(x_gt) if ae is not None else None

        # Stage 5: Gaussian prior sampling in the MLP latent space
        zs = torch.cat([mlp.features(x.to(device)) for x, _ in test_loader])
        mu, sigma = zs.mean(0), zs.std(0)
        z_rand = mu + sigma * torch.randn(16, zs.size(1), device=device)
        x_rand = decoder(z_rand)

        # Quantitative check: does the classifier recognise its own "samples"?
        conf_real = F.softmax(mlp(decoder(zs[:2000])), 1).max(1)[0].mean().item()
        z_many = mu + sigma * torch.randn(2000, zs.size(1), device=device)
        conf_rand = F.softmax(mlp(decoder(z_many)), 1).max(1)[0].mean().item()
        results["stage5_prior_sampling"] = {
            "fraction_negative_in_samples": (z_many < 0).float().mean().item(),
            "fraction_zero_in_real_z": (zs == 0).float().mean().item(),
            "mlp_confidence_on_decoded_real_z": conf_real,
            "mlp_confidence_on_decoded_random_z": conf_rand,
        }

    # ---------------- Figure 3
    rows = [
        (x_gt, "(1) Ground truth (test set): " + ", ".join(CLASSES[c] for c in labels), None),
        (x_class, "(2a) W^T projection from one-hot class vectors y_c (classes 0-9, raw)", None),
        (minmax_per_image(x_class), "(2b) Same as (2a), min-max scaled per image for visibility", None),
        (x_lat_T, "(2c) W^T projection from z_MLP of the ground-truth images", None),
        (x_lat_pinv, "(extra) Pseudo-inverse projection from z_MLP", None),
        (x_frozen, "(3a) Frozen latent decoder from z_MLP", None),
    ]
    if x_ae is not None:
        rows.append((x_ae, "(3b) Vanilla AE reconstruction from z_AE (same latent dim 64)", None))
    rows.append((x_rand, "(4) Random prior samples z ~ N(mu_z, diag(sigma_z^2)) -> frozen decoder", 8))

    heights = [2 if (nrow and imgs.size(0) > nrow) else 1 for imgs, _, nrow in rows]
    fig, axes = plt.subplots(len(rows), 1, figsize=(11, 1.55 * sum(heights) + 0.6),
                             gridspec_kw={"height_ratios": heights})
    for ax, (imgs, title, nrow) in zip(axes, rows):
        show_row(ax, imgs, title, nrow)
    fig.suptitle("Figure 3. Limits of MLP-classifier weights and latent features for image generation",
                 fontsize=12, y=0.995)
    fig.tight_layout()
    out = "outputs/fig3_mlp_limitations.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved {out}")

    save_json(results, "results/mlp_limitations.json")

    # Short console summary
    print("\n=== Summary ===")
    print("Stage 3 W^T recon :", results["stage3_transpose_test_metrics"])
    print("Pseudo-inverse    :", results["extra_pinv_test_metrics"])
    print("Frozen decoder    :", results["stage4_frozen_decoder_test_metrics"])
    if "stage4_vanilla_ae_test_metrics" in results:
        print("Vanilla AE        :", results["stage4_vanilla_ae_test_metrics"])
    print("Prior sampling    :", results["stage5_prior_sampling"])


if __name__ == "__main__":
    main()
