import argparse, time
import torch
import torch.nn.functional as F

from data import get_loaders
from models.mlp import MLPClassifier
from utils import ensure_dir, get_device, save_json, set_seed

CKPT = "checkpoints/mlp_classifier.pt"


@torch.no_grad()
def accuracy(model, loader, device):
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        correct += (model(x).argmax(1) == y).sum().item()
        total += y.numel()
    return correct / total


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    set_seed(args.seed)
    device = get_device()
    ensure_dir("checkpoints")
    train_loader, val_loader, test_loader = get_loaders("01", args.batch_size)

    model = MLPClassifier().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    history = {"train_loss": [], "val_acc": [], "epoch_time_s": []}
    best_val = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.perf_counter()
        total, n = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            loss = F.cross_entropy(model(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * x.size(0)
            n += x.size(0)
        if device.type == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0

        val_acc = accuracy(model, val_loader, device)
        history["train_loss"].append(total / n)
        history["val_acc"].append(val_acc)
        history["epoch_time_s"].append(dt)
        print(f"epoch {epoch:02d}  loss {total / n:.4f}  val_acc {val_acc:.4f}  ({dt:.1f}s)")

        if val_acc > best_val:          # keep the best model on the validation set
            best_val = val_acc
            torch.save(model.state_dict(), CKPT)

    model.load_state_dict(torch.load(CKPT, map_location=device))
    test_acc = accuracy(model, test_loader, device)
    print(f"\nBest val acc {best_val:.4f} | TEST ACC {test_acc:.4f} "
          f"{'(meets >=88%)' if test_acc >= 0.88 else '(BELOW 88%)'}")

    save_json({
        "test_acc": test_acc,
        "best_val_acc": best_val,
        "mean_epoch_time_s": sum(history["epoch_time_s"]) / len(history["epoch_time_s"]),
        "config": vars(args),
        "history": history,
    }, "results/mlp_classifier.json")


if __name__ == "__main__":
    main()
