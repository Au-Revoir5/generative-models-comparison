"""Final evaluation of all six models on the same footing:
  - FID (InceptionV3, torchmetrics) with N generated vs N real TEST images
  - inference latency per batch of 64 (warm-up + CUDA sync, same GPU for all)
  - mode coverage (class distribution of samples, judged by the trained MLP classifier)
  - collects test MSE/PSNR/SSIM and training time per epoch from results/*.json
Writes results/final_metrics.json and results/final_table.md

Run:  python -m evaluation.evaluate_all            (default N = 5000)
DDPM sampling dominates the run time; its samples are cached in
checkpoints/ddpm_samples.pt chunk by chunk, so a disconnect does not lose progress.
"""
import argparse, json, math, os, time

import torch
import torch.nn.functional as F
from torchmetrics.image.fid import FrechetInceptionDistance

from data import get_loaders, to_01
from models.ae import VanillaAE
from models.dcgan import Generator as DCGenerator
from models.ddpm import GaussianDiffusion, UNet
from models.gan import Generator as MLPGenerator
from models.mlp import MLPClassifier
from models.sparse_ae import SparseAE
from models.vae import VAE
from utils import get_device, save_json, set_seed

DDPM_CACHE = "checkpoints/ddpm_samples.pt"


# ------------------------------------------------------------------ helpers
def load(model, path, device):
    model.load_state_dict(torch.load(path, map_location=device))
    return model.to(device).eval()


def read_json(path):
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def collect(loader, n):
    imgs, total = [], 0
    for x, _ in loader:
        imgs.append(x); total += x.size(0)
        if total >= n:
            break
    return torch.cat(imgs)[:n]


@torch.no_grad()
def generate(fn, n, bs=500):
    """fn(b) -> b images in [0,1]. Returns n images on the CPU."""
    out, done = [], 0
    while done < n:
        b = min(bs, n - done)
        out.append(fn(b).float().clamp(0, 1).cpu())
        done += b
    return torch.cat(out)


@torch.no_grad()
def fid_score(fid, fake, device, bs=250):
    fid.reset()                                    # keeps the cached real features
    for i in range(0, len(fake), bs):
        fid.update(fake[i:i + bs].to(device).repeat(1, 3, 1, 1), real=False)
    return fid.compute().item()


@torch.no_grad()
def latency_ms(fn, device, runs=20, warmup=3):
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(runs):
        sync(device)
        t0 = time.perf_counter()
        fn()
        sync(device)
        times.append((time.perf_counter() - t0) * 1000)
    mean = sum(times) / len(times)
    std = math.sqrt(sum((t - mean) ** 2 for t in times) / len(times))
    return {"mean_ms": mean, "std_ms": std, "runs": runs}


@torch.no_grad()
def mode_coverage(clf, imgs, device, bs=500):
    counts = torch.zeros(10)
    conf = 0.0
    for i in range(0, len(imgs), bs):
        prob = F.softmax(clf(imgs[i:i + bs].to(device)), 1).cpu()
        counts += torch.bincount(prob.argmax(1), minlength=10).float()
        conf += prob.max(1)[0].sum().item()
    p = counts / counts.sum()
    ent = -(p[p > 0] * p[p > 0].log()).sum().item()
    return {"class_distribution": p.tolist(), "normalized_entropy": ent / math.log(10),
            "mean_classifier_confidence": conf / len(imgs)}


@torch.no_grad()
def latent_gaussian_sampler(model, loader, device):
    """Extra for AE / Sparse AE: fit N(mu, diag sigma^2) to the encoded test set and
    decode samples from it. Shows how much of that Gaussian falls into latent voids."""
    zs = torch.cat([model.encode(x.to(device)) for x, _ in loader])
    mu, sd = zs.mean(0), zs.std(0)
    return lambda b: model.decode(mu + sd * torch.randn(b, zs.size(1), device=device))


def tail_std(values, frac=0.2):
    """Std of the last `frac` of an iteration-level loss log (oscillation measure)."""
    if not values:
        return None
    tail = values[int(len(values) * (1 - frac)):]
    m = sum(tail) / len(tail)
    return math.sqrt(sum((v - m) ** 2 for v in tail) / len(tail))


# ------------------------------------------------------------------ main
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=5000, help="images per model for FID")
    p.add_argument("--ddpm-bs", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    set_seed(args.seed)
    device = get_device()
    _, val_loader, test_loader = get_loaders("01", 128)

    # ---------------- models
    ae = load(VanillaAE(64), "checkpoints/vanilla_ae.pt", device)
    sae = load(SparseAE(64), "checkpoints/sparse_ae.pt", device)
    vae = load(VAE(64), "checkpoints/vae.pt", device)
    gan = load(MLPGenerator(100), "checkpoints/gan_G.pt", device)
    dcgan = load(DCGenerator(100), "checkpoints/dcgan_G.pt", device)
    ddpm = load(UNet(), "checkpoints/ddpm_ema.pt", device)
    diffusion = GaussianDiffusion(T=1000, device=device)
    clf = load(MLPClassifier(), "checkpoints/mlp_classifier.pt", device)

    samplers = {
        "VAE": lambda b: vae.sample(b, device),
        "Vanilla GAN": lambda b: to_01(gan(torch.randn(b, 100, device=device))),
        "DCGAN": lambda b: to_01(dcgan(torch.randn(b, 100, device=device))),
        "DDPM": lambda b: to_01(diffusion.sample(ddpm, b, device=device)[0]),
        "Vanilla AE (extra: Gaussian-fit latent)": latent_gaussian_sampler(ae, test_loader, device),
        "Sparse AE (extra: Gaussian-fit latent)": latent_gaussian_sampler(sae, test_loader, device),
    }

    # ---------------- FID setup (real = test set)
    fid = FrechetInceptionDistance(feature=2048, normalize=True, reset_real_features=False).to(device)
    real = collect(test_loader, args.n)
    for i in range(0, len(real), 250):
        fid.update(real[i:i + 250].to(device).repeat(1, 3, 1, 1), real=True)
    print(f"Real test features computed ({len(real)} images)")

    out = {"n_fid": args.n, "fid": {}, "mode_coverage": {}, "latency_batch64": {}}

    # Reference: FID between two sets of REAL images (val vs test) = best achievable
    out["fid"]["Real val images (reference floor)"] = fid_score(fid, collect(val_loader, args.n), device)
    print(f"FID real-vs-real reference: {out['fid']['Real val images (reference floor)']:.2f}")

    for name, fn in samplers.items():
        torch.manual_seed(args.seed)
        if name == "DDPM":
            fake = torch.load(DDPM_CACHE) if os.path.exists(DDPM_CACHE) else torch.empty(0, 1, 28, 28)
            chunk = len(fake) // args.ddpm_bs
            while len(fake) < args.n:
                torch.manual_seed(args.seed + chunk)
                t0 = time.perf_counter()
                b = min(args.ddpm_bs, args.n - len(fake))
                fake = torch.cat([fake, generate(fn, b, bs=b)])
                torch.save(fake, DDPM_CACHE)
                chunk += 1
                print(f"  DDPM samples {len(fake)}/{args.n}  ({time.perf_counter() - t0:.0f}s for this chunk)")
            fake = fake[:args.n]
        else:
            fake = generate(fn, args.n)
        out["fid"][name] = fid_score(fid, fake, device)
        out["mode_coverage"][name] = mode_coverage(clf, fake, device)
        print(f"FID {name}: {out['fid'][name]:.2f}  |  coverage entropy "
              f"{out['mode_coverage'][name]['normalized_entropy']:.3f}")

    # ---------------- latency per batch of 64 (same GPU, warm-up, CUDA sync)
    x64 = collect(test_loader, 64).to(device)
    z64 = torch.randn(64, 100, device=device)
    lat_fns = {
        "Vanilla AE (reconstruct)": (lambda: ae(x64), 50),
        "Sparse AE (reconstruct)": (lambda: sae(x64), 50),
        "VAE (reconstruct)": (lambda: vae.reconstruct(x64), 50),
        "VAE (sample from prior)": (lambda: vae.sample(64, device), 50),
        "Vanilla GAN (generate)": (lambda: gan(z64), 50),
        "DCGAN (generate)": (lambda: dcgan(z64), 50),
        "DDPM (generate, 1000 steps)": (lambda: diffusion.sample(ddpm, 64, device=device), 3),
    }
    for name, (fn, runs) in lat_fns.items():
        out["latency_batch64"][name] = latency_ms(fn, device, runs=runs, warmup=1 if runs < 10 else 3)
        print(f"Latency {name}: {out['latency_batch64'][name]['mean_ms']:.2f} ms")

    # ---------------- collect training results
    R = {k: read_json(f"results/{k}.json") for k in ("vanilla_ae", "sparse_ae", "vae", "gan", "dcgan", "ddpm")}
    out["stability"] = {
        "Vanilla GAN loss_G std (last 20% iters)": tail_std(R["gan"].get("iter_history", {}).get("loss_G", [])),
        "Vanilla GAN loss_D std (last 20% iters)": tail_std(R["gan"].get("iter_history", {}).get("loss_D", [])),
        "DCGAN loss_G std (last 20% iters)": tail_std(R["dcgan"].get("iter_history", {}).get("loss_G", [])),
        "DCGAN loss_D std (last 20% iters)": tail_std(R["dcgan"].get("iter_history", {}).get("loss_D", [])),
        "DDPM loss std (last 20% iters)": tail_std(R["ddpm"].get("iter_loss", [])),
    }

    def fmt(v, d=4):
        return "N/A" if v is None else f"{v:.{d}f}"

    rows = [
        ("Vanilla AE", "64", R["vanilla_ae"], "Vanilla AE (reconstruct)", None, "Stable / converged"),
        ("Sparse AE (λ=1e-3)", "64 (bottleneck) / 256 (sparse)", R["sparse_ae"], "Sparse AE (reconstruct)", None, "Stable / converged"),
        ("VAE", "64", R["vae"], "VAE (sample from prior)", "VAE", "Stable / converged"),
        ("Vanilla GAN", "100 (prior noise)", R["gan"], "Vanilla GAN (generate)", "Vanilla GAN", "Adversarial; see loss std"),
        ("DCGAN", "100 (prior noise)", R["dcgan"], "DCGAN (generate)", "DCGAN", "Adversarial; see loss std"),
        ("DDPM", "28x28 (image dim)", R["ddpm"], "DDPM (generate, 1000 steps)", "DDPM", "Very stable (MSE regression)"),
    ]
    lines = ["| Model | Latent dim | Train time / epoch (s) | Latency (ms / batch 64) | Test MSE | Test PSNR (dB) | Test SSIM | FID | Coverage entropy | Training stability |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for name, dim, res, lat_key, fid_key, stab in rows:
        test = res.get("test", {})
        lines.append("| " + " | ".join([
            name, dim, fmt(res.get("mean_epoch_time_s"), 1),
            fmt(out["latency_batch64"][lat_key]["mean_ms"], 2),
            fmt(test.get("mse"), 5), fmt(test.get("psnr"), 2), fmt(test.get("ssim"), 4),
            fmt(out["fid"].get(fid_key) if fid_key else None, 2),
            fmt(out["mode_coverage"][fid_key]["normalized_entropy"] if fid_key else None, 3),
            stab]) + " |")
    lines += ["", f"FID computed with {args.n} generated vs {args.n} real test images (InceptionV3 pool features, 2048-d).",
              f"Real-vs-real reference FID (val vs test): {out['fid']['Real val images (reference floor)']:.2f}",
              "", "Extra (not in template): FID when sampling a Gaussian fitted to the AE latents:",
              f"- Vanilla AE: {out['fid']['Vanilla AE (extra: Gaussian-fit latent)']:.2f}",
              f"- Sparse AE: {out['fid']['Sparse AE (extra: Gaussian-fit latent)']:.2f}",
              "", "Loss oscillation (std of per-iteration loss, last 20% of training):"]
    lines += [f"- {k}: {fmt(v, 4)}" for k, v in out["stability"].items()]
    table = "\n".join(lines)

    with open("results/final_table.md", "w", encoding="utf-8") as f:
        f.write(table + "\n")
    save_json(out, "results/final_metrics.json")
    print("\n" + table)


if __name__ == "__main__":
    main()
