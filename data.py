import torch
from torch.utils.data import ConcatDataset, DataLoader, random_split
from torchvision import datasets, transforms

DATA_ROOT = "/content/data"          # local disk: fast, re-downloads each session (~30 MB)
SPLIT_SEED = 42
SPLIT_SIZES = (56_000, 7_000, 7_000)  # 80/10/10 of 70k

CLASSES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
           "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]


def get_transform(pixel_range="01"):
    if pixel_range == "01":                     # AE, Sparse AE, VAE, MLP
        return transforms.ToTensor()
    if pixel_range == "-11":                    # GAN, DCGAN, DDPM
        return transforms.Compose([transforms.ToTensor(),
                                   transforms.Normalize((0.5,), (0.5,))])
    raise ValueError(pixel_range)


def get_datasets(pixel_range="01"):
    tf = get_transform(pixel_range)
    full = ConcatDataset([
        datasets.FashionMNIST(DATA_ROOT, train=True, download=True, transform=tf),
        datasets.FashionMNIST(DATA_ROOT, train=False, download=True, transform=tf),
    ])
    g = torch.Generator().manual_seed(SPLIT_SEED)
    return random_split(full, SPLIT_SIZES, generator=g)


def get_loaders(pixel_range="01", batch_size=128, num_workers=2):
    train_set, val_set, test_set = get_datasets(pixel_range)
    kw = dict(batch_size=batch_size, num_workers=num_workers,
              pin_memory=torch.cuda.is_available())
    return (DataLoader(train_set, shuffle=True, **kw),
            DataLoader(val_set, shuffle=False, **kw),
            DataLoader(test_set, shuffle=False, **kw))


def to_01(x):
    """[-1,1] -> [0,1], used before metrics and plots for GAN/DDPM outputs."""
    return ((x + 1) / 2).clamp(0, 1)


if __name__ == "__main__":
    for rng in ("01", "-11"):
        tr, va, te = get_loaders(rng)
        x, y = next(iter(tr))
        print(f"[{rng}] {tuple(x.shape)} min={x.min():.2f} max={x.max():.2f} "
              f"| train={len(tr.dataset)} val={len(va.dataset)} test={len(te.dataset)}")
