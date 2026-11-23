import argparse, os, time
import torch
import torch.nn.functional as F

from data import get_loaders
from evaluation.metrics import reconstruction_metrics
from models.ae import VanillaAE
from models.sparse_ae import SparseAE
from utils import ensure_dir, get_device, save_json, set_seed


@torch.no_grad()
def eval_loss(model, loader, device, lam):
    model.eval()
    mse_t = l1_t = 0.0; n = 0
    for x, _ in loader:
        x = x.to(device)
        x_hat, h = model(x)
        mse_t += F.mse_loss(x_hat, x).item() * x.size(0)
        l1_t += h.abs().mean().item() * x.size(0)
        n += x.size(0)
    return mse_t / n + lam * l1_t / n, mse_t / n


@torch.no_grad()
def sparsity_stats(hidden_fn, loader, device, thr=1e-2):
    """How sparse the 256-unit hidden code is on the test set."""
    zeros = small = total = 0; abs_sum = 0.0; n = 0
    for x, _ in loader:
        h = hidden_fn(x.to(device))
        zeros += (h == 0).sum().item()
        small += (h.abs() < thr).sum().item()
        abs_sum += h.abs().sum().item()
        total += h.numel(); n += h.size(0)
    return {"exact_zero_fraction": zeros / total,
            f"below_{thr}_fraction": small / total,
            "mean_abs_activation": abs_sum / total,
            "avg_active_units_per_image": (total - small) / n}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lam", type=float, default=1e-3)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--latent-dim", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--name", default="sparse_ae")   # use a different name for lambda sweeps
    args = p.parse_args()

    set_seed(args.seed)
    device = get_device()
    ensure_dir("checkpoints")
    ckpt = f"checkpoints/{args.name}.pt"
    train_loader, val_loader, test_loader = get_loaders("01", args.batch_size)

    model = SparseAE(latent_dim=args.latent_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    hist = {"train_loss": [], "train_mse": [], "train_l1": [], "val_loss": [], "val_mse": [], "epoch_time_s": []}
    best = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.perf_counter()
        tl = tm = t1 = 0.0; n = 0
        for x, _ in train_loader:
            x = x.to(device)
            x_hat, h = model(x)
            mse = F.mse_loss(x_hat, x)
            l1 = h.abs().mean()
            loss = mse + args.lam * l1
            opt.zero_grad(); loss.backward(); opt.step()
            b = x.size(0)
            tl += loss.item() * b; tm += mse.item() * b; t1 += l1.item() * b; n += b
        if device.type == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0

        v_loss, v_mse = eval_loss(model, val_loader, device, args.lam)
        for k, v in (("train_loss", tl / n), ("train_mse", tm / n), ("train_l1", t1 / n),
                     ("val_loss", v_loss), ("val_mse", v_mse), ("epoch_time_s", dt)):
            hist[k].append(v)
        print(f"epoch {epoch:02d}  mse {tm/n:.5f}  l1 {t1/n:.4f}  val_loss {v_loss:.5f}  ({dt:.1f}s)")
        if v_loss < best:
            best = v_loss
            torch.save(model.state_dict(), ckpt)

    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    test = reconstruction_metrics(lambda x: model(x)[0], test_loader, device)
    sparsity = sparsity_stats(model.hidden, test_loader, device)
    print(f"\nTEST  MSE {test['mse']:.5f}  PSNR {test['psnr']:.2f} dB  SSIM {test['ssim']:.4f}")
    print("Sparse AE hidden code:", sparsity)

    out = {"lambda": args.lam, "test": test, "sparsity": sparsity,
           "mean_epoch_time_s": sum(hist["epoch_time_s"]) / len(hist["epoch_time_s"]),
           "config": vars(args), "history": hist}

    # Same sparsity stats for the Vanilla AE hidden layer, for the Q2 comparison
    if os.path.exists("checkpoints/vanilla_ae.pt"):
        ae = VanillaAE(args.latent_dim).to(device)
        ae.load_state_dict(torch.load("checkpoints/vanilla_ae.pt", map_location=device))
        ae.eval()
        out["vanilla_ae_hidden_sparsity"] = sparsity_stats(lambda x: ae.encoder[:3](x), test_loader, device)
        print("Vanilla AE hidden code:", out["vanilla_ae_hidden_sparsity"])

    save_json(out, f"results/{args.name}.json")


if __name__ == "__main__":
    main()
