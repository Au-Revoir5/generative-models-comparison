import argparse, copy, os, time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torchvision.utils import save_image

from data import get_loaders, to_01
from models.ddpm import GaussianDiffusion, UNet
from utils import ensure_dir, get_device, save_json, set_seed

LAST = "checkpoints/ddpm_last.pt"      # full training state (for resuming)
EMA_CKPT = "checkpoints/ddpm_ema.pt"   # EMA weights used for sampling and evaluation


@torch.no_grad()
def ema_update(ema, model, decay):
    for pe, p in zip(ema.parameters(), model.parameters()):
        pe.mul_(decay).add_(p.detach(), alpha=1 - decay)


@torch.no_grad()
def val_loss(model, diffusion, loader, device):
    """Noise-prediction MSE on the val set with a fixed generator (comparable across epochs)."""
    model.eval()
    g = torch.Generator(device=device).manual_seed(0)
    total, n = 0.0, 0
    for x, _ in loader:
        x = x.to(device)
        total += diffusion.loss(model, x, generator=g).item() * x.size(0)
        n += x.size(0)
    return total / n


def save_trajectory(snaps, path, n_rows=6):
    ts = sorted(snaps.keys(), reverse=True)
    fig, axes = plt.subplots(n_rows, len(ts), figsize=(1.3 * len(ts), 1.3 * n_rows + 0.4))
    for j, t in enumerate(ts):
        imgs = to_01(snaps[t][:n_rows]).cpu()
        for i in range(n_rows):
            axes[i, j].imshow(imgs[i, 0], cmap="gray", vmin=0, vmax=1)
            axes[i, j].axis("off")
        axes[0, j].set_title(f"t={t}" if t < max(ts) else f"t={t}\n(x_T noise)", fontsize=9)
    fig.suptitle("DDPM reverse diffusion trajectory  x_T -> x_0", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--T", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--ema-decay", type=float, default=0.999)
    p.add_argument("--sample-every", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    set_seed(args.seed)
    device = get_device()
    use_amp = device.type == "cuda"
    ensure_dir("checkpoints"); ensure_dir("outputs")
    train_loader, val_loader, _ = get_loaders("-11", args.batch_size)

    diffusion = GaussianDiffusion(T=args.T, device=device)
    model = UNet().to(device)
    ema = copy.deepcopy(model).eval()
    for p_ in ema.parameters():
        p_.requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    print(f"U-Net parameters: {sum(p_.numel() for p_ in model.parameters()) / 1e6:.2f} M")

    start_epoch, step = 1, 0
    hist = {"train_loss": [], "val_loss": [], "epoch_time_s": [], "iter_loss": []}
    if os.path.exists(LAST):
        state = torch.load(LAST, map_location=device)
        model.load_state_dict(state["model"]); ema.load_state_dict(state["ema"])
        opt.load_state_dict(state["opt"]); scaler.load_state_dict(state["scaler"])
        start_epoch, step, hist = state["epoch"] + 1, state["step"], state["hist"]
        print(f"Resumed from epoch {state['epoch']}")

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        t0 = time.perf_counter()
        total, n = 0.0, 0
        for x, _ in train_loader:
            x = x.to(device)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                loss = diffusion.loss(model, x)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            step += 1
            ema_update(ema, model, min(args.ema_decay, (1 + step) / (10 + step)))
            total += loss.item() * x.size(0); n += x.size(0)
            hist["iter_loss"].append(loss.item())
        if device.type == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0

        v = val_loss(ema, diffusion, val_loader, device)
        hist["train_loss"].append(total / n); hist["val_loss"].append(v); hist["epoch_time_s"].append(dt)
        print(f"epoch {epoch:02d}  train_mse {total / n:.4f}  val_mse(ema) {v:.4f}  ({dt:.1f}s)")

        torch.save({"model": model.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(),
                    "scaler": scaler.state_dict(), "epoch": epoch, "step": step, "hist": hist}, LAST)
        torch.save(ema.state_dict(), EMA_CKPT)

        if epoch % args.sample_every == 0 and epoch != args.epochs:
            torch.manual_seed(args.seed)
            x0, _ = diffusion.sample(ema, 64, device=device)
            save_image(to_01(x0), f"outputs/ddpm_epoch{epoch:03d}.png", nrow=8)
            print(f"  saved outputs/ddpm_epoch{epoch:03d}.png")

    # Final samples + reverse trajectory for Figure 5
    torch.manual_seed(args.seed)
    snap_ts = (800, 600, 400, 200, 100, 50, 0)
    t0 = time.perf_counter()
    x0, snaps = diffusion.sample(ema, 64, device=device, snapshot_ts=snap_ts)
    if device.type == "cuda":
        torch.cuda.synchronize()
    print(f"Sampling 64 images took {time.perf_counter() - t0:.1f}s ({args.T} steps)")
    save_image(to_01(x0), "outputs/ddpm_final.png", nrow=8)
    save_trajectory(snaps, "outputs/ddpm_trajectory.png")
    print("Saved outputs/ddpm_final.png and outputs/ddpm_trajectory.png")

    save_json({"mean_epoch_time_s": sum(hist["epoch_time_s"]) / len(hist["epoch_time_s"]),
               "final_val_mse": hist["val_loss"][-1], "config": vars(args),
               "history": {k: hist[k] for k in ("train_loss", "val_loss", "epoch_time_s")},
               "iter_loss": hist["iter_loss"]}, "results/ddpm.json")


if __name__ == "__main__":
    main()
