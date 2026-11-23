import argparse, time
import torch
import torch.nn.functional as F

from data import get_loaders
from evaluation.metrics import reconstruction_metrics
from models.ae import VanillaAE
from utils import ensure_dir, get_device, save_json, set_seed

CKPT = "checkpoints/vanilla_ae.pt"


@torch.no_grad()
def eval_loss(model, loader, device):
    model.eval()
    total, n = 0.0, 0
    for x, _ in loader:
        x = x.to(device)
        total += F.mse_loss(model(x), x).item() * x.size(0)
        n += x.size(0)
    return total / n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--latent-dim", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    set_seed(args.seed)
    device = get_device()
    ensure_dir("checkpoints")
    train_loader, val_loader, test_loader = get_loaders("01", args.batch_size)

    model = VanillaAE(latent_dim=args.latent_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    history = {"train_loss": [], "val_loss": [], "epoch_time_s": []}
    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.perf_counter()
        total, n = 0.0, 0
        for x, _ in train_loader:
            x = x.to(device)
            loss = F.mse_loss(model(x), x)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * x.size(0)
            n += x.size(0)
        if device.type == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0

        val_loss = eval_loss(model, val_loader, device)
        history["train_loss"].append(total / n)
        history["val_loss"].append(val_loss)
        history["epoch_time_s"].append(dt)
        print(f"epoch {epoch:02d}  train_mse {total / n:.5f}  val_mse {val_loss:.5f}  ({dt:.1f}s)")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), CKPT)

    model.load_state_dict(torch.load(CKPT, map_location=device))
    model.eval()
    test = reconstruction_metrics(model, test_loader, device)
    print(f"\nTEST  MSE {test['mse']:.5f}  PSNR {test['psnr']:.2f} dB  SSIM {test['ssim']:.4f}")

    save_json({
        "test": test,
        "best_val_mse": best_val,
        "mean_epoch_time_s": sum(history["epoch_time_s"]) / len(history["epoch_time_s"]),
        "config": vars(args),
        "history": history,
    }, "results/vanilla_ae.json")


if __name__ == "__main__":
    main()
