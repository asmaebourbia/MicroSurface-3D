"""Procedural periodic Gaussian microtopographies, independent of external data."""
from pathlib import Path
import argparse
import numpy as np
from scipy.ndimage import gaussian_filter
from torch.utils.data import Dataset
from .physics import stats_np


def synthetic_surface(size, rng):
    """A random anisotropic correlated height field with controlled RMS slope.

    Pixel spacing = 1 in arbitrary units. No real manufacturing physics.
    """
    noise = rng.standard_normal((size, size)).astype(np.float32)
    sigmax = rng.uniform(1.4, 5.0)
    sigmay = rng.uniform(1.4, 5.0)
    h = gaussian_filter(noise, sigma=(sigmay, sigmax), mode='wrap')
    # Add occasional multiscale component without inheriting an SR architecture.
    if rng.random() < 0.5:
        h += 0.25 * gaussian_filter(noise, sigma=(max(0.8, sigmay/2),
                                                     max(0.8, sigmax/2)), mode='wrap')
    h -= h.mean()
    target_slope = rng.uniform(0.08, 0.28)
    current = float(stats_np(h)[0])
    h *= target_slope / max(current, 1e-8)
    return h.astype(np.float32)


def generate_dataset(path, n_train=2400, n_val=300, size=64, seed=43):
    if size != 64:
        raise ValueError('This starter GAN expects size 64 (change the network for others).')
    rng = np.random.default_rng(seed)
    n = n_train + n_val
    h = np.empty((n, size, size), dtype=np.float32)
    c = np.empty((n, 2), dtype=np.float32)
    for i in range(n):
        h[i] = synthetic_surface(size, rng)
        c[i] = stats_np(h[i])
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, train_h=h[:n_train], train_c=c[:n_train],
                        val_h=h[n_train:], val_c=c[n_train:],
                        seed=np.int32(seed), grid_spacing=np.float32(1))
    print(f'Saved {path} | train={n_train} val={n_val} size={size}')
    print(f'RMS slope range: {c[:,0].min():.3f}–{c[:,0].max():.3f}')
    print(f'Anisotropy ratio range: {c[:,1].min():.3f}–{c[:,1].max():.3f}')


class Surfaces(Dataset):
    def __init__(self, filename, split='train'):
        with np.load(filename) as data:
            self.h = data[f'{split}_h'].copy()
            self.c = data[f'{split}_c'].copy()
    def __len__(self):
        return len(self.h)
    def __getitem__(self, i):
        return self.h[i][None], self.c[i]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', default='data/surfaces.npz')
    p.add_argument('--train', type=int, default=2400)
    p.add_argument('--val', type=int, default=300)
    p.add_argument('--size', type=int, default=64)
    p.add_argument('--seed', type=int, default=43)
    a = p.parse_args()
    generate_dataset(a.output, a.train, a.val, a.size, a.seed)

if __name__ == '__main__':
    main()
