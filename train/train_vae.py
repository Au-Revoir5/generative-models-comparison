import argparse, time
import torch
import torch.nn.functional as F
from torchvision.utils import save_image

from data import get_loaders
from evaluation.metrics import reconstruction_metrics
from models.vae import VAE
from utils import ensure_dir, get_device, save_json, set_seed

CKPT = "checkpoints/vae.pt"


def vae_loss(x_hat, x, mu, logvar, beta=1.0):
    b = x.size(0)
    recon = F.binary_cross_entropy(x_hat, x, reduction="sum") / b
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / b
    return recon + beta * kl, recon, kl


@torch.no_grad()
def eval_loss(model, loader, device, beta):
    model.eval()
    t = r = k = 0.0; n = 0
    for x, _ in loader:
        x = x.to(device)
        x_hat, mu, logvar = model(x)
        loss, rec, kl = vae_loss(x_hat, x, mu, logvar, beta)
        b = x.size(0)
        t += loss.item() * b; r += rec.item() * b; k += kl.item() * b; n += b
    return t / n, r / n, k / n


@torch.no_grad()
def active_latent_dims(model, loader, device, thr=0.01):
    """Latent dims whose average KL exceeds thr; the rest collapsed to the prior."""
    model.eval()
    kl_sum = None; n = 0
    for x, _ in loader:
        mu, logvar = model.encode(x.to(device))
        kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        kl_sum = kl.sum(0) if kl_sum is None else kl_sum + kl.sum(0)
        n += x.size(0)
    kl_per_dim = kl_sum / n
    return int((kl_per_dim > thr).sum().item()), kl_per_dim.tolist()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--latent-dim", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    set_seed(args.seed)
    device = get_device()
    ensure_dir("checkpoints"); ensure_dir("outputs")
    train_loader, val_loader, test_loader = get_loaders("01", args.batch_size)

    model = VAE(latent_dim=args.latent_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    hist = {"train_loss": [], "train_recon": [], "train_kl": [], "val_loss": [], "epoch_time_s": []}
    best = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.perf_counter()
        t = r = k = 0.0; n = 0
        for x, _ in train_loader:
            x = x.to(device)
            x_hat, mu, logvar = model(x)
            loss, rec, kl = vae_loss(x_hat, x, mu, logvar, args.beta)
            opt.zero_grad(); loss.backward(); opt.step()
            b = x.size(0)
            t += loss.item() * b; r += rec.item() * b; k += kl.item() * b; n += b
        if device.type == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0

        v_loss, _, _ = eval_loss(model, val_loader, device, args.beta)
        for key, v in (("train_loss", t / n), ("train_recon", r / n), ("train_kl", k / n),
                       ("val_loss", v_loss), ("epoch_time_s", dt)):
            hist[key].append(v)
        print(f"epoch {epoch:02d}  loss {t/n:.2f}  recon {r/n:.2f}  kl {k/n:.2f}  val {v_loss:.2f}  ({dt:.1f}s)")
        if v_loss < best:
            best = v_loss
            torch.save(model.state_dict(), CKPT)

    model.load_state_dict(torch.load(CKPT, map_location=device))
    model.eval()
    test = reconstruction_metrics(model.reconstruct, test_loader, device)
    n_active, kl_per_dim = active_latent_dims(model, test_loader, device)
    print(f"\nTEST  MSE {test['mse']:.5f}  PSNR {test['psnr']:.2f} dB  SSIM {test['ssim']:.4f}")
    print(f"Active latent dims: {n_active} / {args.latent_dim}")

    torch.manual_seed(args.seed)
    save_image(model.sample(64, device), "outputs/vae_samples_preview.png", nrow=8)
    print("Saved outputs/vae_samples_preview.png")

    save_json({"test": test, "active_latent_dims": n_active, "kl_per_dim": kl_per_dim,
               "mean_epoch_time_s": sum(hist["epoch_time_s"]) / len(hist["epoch_time_s"]),
               "config": vars(args), "history": hist}, "results/vae.json")


if __name__ == "__main__":
    main()
