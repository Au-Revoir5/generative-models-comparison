| Model | Latent dim | Train time / epoch (s) | Latency (ms / batch 64) | Test MSE | Test PSNR (dB) | Test SSIM | FID | Coverage entropy | Training stability |
|---|---|---|---|---|---|---|---|---|---|
| Vanilla AE | 64 | 7.4 | 0.35 | 0.00650 | 23.00 | 0.8667 | N/A | N/A | Stable / converged |
| Sparse AE (λ=1e-3) | 64 (bottleneck) / 256 (sparse) | 7.3 | 0.23 | 0.00673 | 22.84 | 0.8629 | N/A | N/A | Stable / converged |
| VAE | 64 | 7.1 | 0.18 | 0.01310 | 19.68 | 0.7683 | 85.56 | 0.983 | Stable / converged |
| Vanilla GAN | 100 (prior noise) | 12.8 | 0.23 | N/A | N/A | N/A | 127.05 | 0.995 | Adversarial; see loss std |
| DCGAN | 100 (prior noise) | 27.5 | 10.93 | N/A | N/A | N/A | 18.21 | 0.997 | Adversarial; see loss std |
| DDPM | 28x28 (image dim) | 70.5 | 47936.61 | N/A | N/A | N/A | 12.12 | 0.992 | Very stable (MSE regression) |

FID computed with 5000 generated vs 5000 real test images (InceptionV3 pool features, 2048-d).
Real-vs-real reference FID (val vs test): 5.24

Extra (not in template): FID when sampling a Gaussian fitted to the AE latents:
- Vanilla AE: 209.38
- Sparse AE: 193.38

Loss oscillation (std of per-iteration loss, last 20% of training):
- Vanilla GAN loss_G std (last 20% iters): 0.0259
- Vanilla GAN loss_D std (last 20% iters): 0.0259
- DCGAN loss_G std (last 20% iters): 0.0754
- DCGAN loss_D std (last 20% iters): 0.0386
- DDPM loss std (last 20% iters): 0.0061
