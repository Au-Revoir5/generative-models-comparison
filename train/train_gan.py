import argparse, math, time
import torch
import torch.nn.functional as F
from torchvision.utils import save_image

from data import get_loaders, to_01
from models.mlp import MLPClassifier
from utils import ensure_dir, get_device, save_json, set_seed


def build(arch, z_dim):
    if arch == "mlp":
        from models.gan import Discriminator, Generator
        return Generator(z_dim), Discriminator(), "gan"
    if arch == "dcgan":
        from models.dcgan import Discriminator, Generator, init_weights
        G, D = Generator(z_dim), Discriminator()
        G.apply(init_weights); D.apply(init_weights)
        return G, D, "dcgan"
    raise ValueError(arch)


@torch.no_grad()
def mode_coverage(G, z_dim, device, n=10_000):
    """Classify n generated samples with the trained MLP classifier.
    A collapsed GAN puts most samples into a few classes (low entropy)."""
    clf = MLPClassifier().to(device)
    clf.load_state_dict(torch.load("checkpoints/mlp_classifier.pt", map_location=device))
    clf.eval(); G.eval()
    counts = torch.zeros(10, device=device)
    conf_sum = 0.0
    for i in range(0, n, 500):
        x = to_01(G(torch.randn(500, z_dim, device=device)))      # classifier expects [0,1]
        prob = F.softmax(clf(x), 1)
        counts += torch.bincount(prob.argmax(1), minlength=10).float()
        conf_sum += prob.max(1)[0].sum().item()
    p = counts / counts.sum()
    entropy = -(p[p > 0] * p[p > 0].log()).sum().item()
    return {"class_distribution": p.tolist(),
            "normalized_entropy": entropy / math.log(10),   # 1.0 = all 10 classes equally
            "mean_classifier_confidence": conf_sum / n}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arch", choices=["mlp", "dcgan"], default="mlp")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--z-dim", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    set_seed(args.seed)
    device = get_device()
    ensure_dir("checkpoints"); ensure_dir("outputs")
    train_loader, _, _ = get_loaders("-11", args.batch_size)

    G, D, name = build(args.arch, args.z_dim)
    G, D = G.to(device), D.to(device)
    opt_G = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(0.5, 0.999))
    opt_D = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(0.5, 0.999))
    bce = F.binary_cross_entropy_with_logits

    fixed_z = torch.randn(64, args.z_dim, device=device)      # same noise every snapshot
    snapshot_epochs = {1, 5, 10, 25, args.epochs}
    it_log = {"loss_D": [], "loss_G": [], "D_real": [], "D_fake": []}   # per iteration
    ep_log = {"loss_D": [], "loss_G": [], "D_real": [], "D_fake": [], "epoch_time_s": []}

    for epoch in range(1, args.epochs + 1):
        G.train(); D.train()
        t0 = time.perf_counter()
        start = len(it_log["loss_D"])
        for x, _ in train_loader:
            x = x.to(device)
            b = x.size(0)
            ones = torch.ones(b, device=device)
            zeros = torch.zeros(b, device=device)

            # --- Discriminator step
            x_fake = G(torch.randn(b, args.z_dim, device=device))
            d_real = D(x)
            d_fake = D(x_fake.detach())
            loss_D = bce(d_real, ones) + bce(d_fake, zeros)
            opt_D.zero_grad(); loss_D.backward(); opt_D.step()

            # --- Generator step (non-saturating: label fakes as real)
            loss_G = bce(D(x_fake), ones)
            opt_G.zero_grad(); loss_G.backward(); opt_G.step()

            it_log["loss_D"].append(loss_D.item())
            it_log["loss_G"].append(loss_G.item())
            it_log["D_real"].append(torch.sigmoid(d_real).mean().item())
            it_log["D_fake"].append(torch.sigmoid(d_fake).mean().item())
        if device.type == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0

        for k in ("loss_D", "loss_G", "D_real", "D_fake"):
            vals = it_log[k][start:]
            ep_log[k].append(sum(vals) / len(vals))
        ep_log["epoch_time_s"].append(dt)
        print(f"epoch {epoch:02d}  loss_D {ep_log['loss_D'][-1]:.3f}  loss_G {ep_log['loss_G'][-1]:.3f}  "
              f"D(x) {ep_log['D_real'][-1]:.2f}  D(G(z)) {ep_log['D_fake'][-1]:.2f}  ({dt:.1f}s)")

        if epoch in snapshot_epochs:
            G.eval()
            with torch.no_grad():
                save_image(to_01(G(fixed_z)), f"outputs/{name}_epoch{epoch:03d}.png", nrow=8)
            G.train()

    torch.save(G.state_dict(), f"checkpoints/{name}_G.pt")
    torch.save(D.state_dict(), f"checkpoints/{name}_D.pt")

    coverage = mode_coverage(G, args.z_dim, device)
    print("\nMode coverage:", {k: (round(v, 3) if isinstance(v, float) else [round(q, 3) for q in v])
                               for k, v in coverage.items()})

    save_json({"mode_coverage": coverage,
               "mean_epoch_time_s": sum(ep_log["epoch_time_s"]) / len(ep_log["epoch_time_s"]),
               "config": vars(args), "epoch_history": ep_log, "iter_history": it_log},
              f"results/{name}.json")


if __name__ == "__main__":
    main()
