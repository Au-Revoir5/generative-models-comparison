"""Reconstruction metrics (MSE, PSNR, SSIM), per image, on [0, 1] tensors."""
import torch
from torchmetrics.functional import structural_similarity_index_measure as ssim_fn


@torch.no_grad()
def reconstruction_metrics(recon_fn, loader, device):
    """recon_fn maps x in [0,1] (B,1,28,28) to its reconstruction in [0,1]."""
    n = 0
    mse_sum = psnr_sum = ssim_sum = 0.0
    for x, _ in loader:
        x = x.to(device)
        x_hat = recon_fn(x).clamp(0, 1)
        mse = ((x_hat - x) ** 2).flatten(1).mean(dim=1)          # per image
        psnr = 10 * torch.log10(1.0 / mse.clamp_min(1e-10))      # data range = 1
        ssim = ssim_fn(x_hat, x, data_range=1.0)                 # batch mean
        b = x.size(0)
        mse_sum += mse.sum().item()
        psnr_sum += psnr.sum().item()
        ssim_sum += ssim.item() * b
        n += b
    return {"mse": mse_sum / n, "psnr": psnr_sum / n, "ssim": ssim_sum / n}
