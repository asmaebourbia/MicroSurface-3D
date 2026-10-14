"""Train a physics-regularized conditional hinge GAN on synthetic height fields."""
import argparse
import csv
import os
import random
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from .data import Surfaces, generate_dataset
from .models import Generator, Discriminator, Z_DIM, weights_init
from .physics import stats_torch


def physical_loss(fake_h, target_cond):
    measured = stats_torch(fake_h)
    # Differentiable RMS slope and directional-anisotropy consistency.
    rough_err = (measured[:, 0] - target_cond[:, 0]).abs() / 0.2
    ratio_err = (measured[:, 1].clamp(min=1e-5).log() -
                 target_cond[:, 1].clamp(min=1e-5).log()).abs()
    return (rough_err + ratio_err).mean()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train(args):
    set_seed(args.seed)
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable: check job GPU allocation and PyTorch CUDA build.')
    device = torch.device('cuda' if (args.device == 'auto' and torch.cuda.is_available()) else
                          ('cpu' if args.device == 'auto' else args.device))
    if device.type == 'cuda':
        print('GPU:', torch.cuda.get_device_name(0), flush=True)
    else:
        print('WARNING: training on CPU', flush=True)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    dataset_file = Path(args.data)
    if not dataset_file.exists():
        generate_dataset(dataset_file, args.num_train, args.num_val, 64, args.seed)
    ds = Surfaces(dataset_file, 'train')
    if not len(ds):
        raise ValueError('Empty training dataset')
    loader = DataLoader(ds, batch_size=args.batch, shuffle=True, drop_last=False,
                        num_workers=args.workers, pin_memory=device.type == 'cuda')
    g = Generator().to(device); d = Discriminator().to(device)
    g.apply(weights_init); d.apply(weights_init)
    opt_g = torch.optim.Adam(g.parameters(), lr=args.lr, betas=(0.0, 0.9))
    opt_d = torch.optim.Adam(d.parameters(), lr=args.lr, betas=(0.0, 0.9))
    start_epoch = 1
    history_file = out / 'history.csv'
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        g.load_state_dict(ckpt['generator']); d.load_state_dict(ckpt['discriminator'])
        opt_g.load_state_dict(ckpt['optim_g']); opt_d.load_state_dict(ckpt['optim_d'])
        start_epoch = ckpt['epoch'] + 1
    if not history_file.exists() or not args.resume:
        with history_file.open('w', newline='') as f:
            csv.writer(f).writerow(['epoch', 'd_loss', 'g_loss', 'physical_loss'])
    for epoch in range(start_epoch, args.epochs + 1):
        g.train(); d.train()
        running = np.zeros(3, dtype=np.float64); batches = 0
        for h, c in loader:
            h = h.to(device=device, dtype=torch.float32, non_blocking=True)
            c = c.to(device=device, dtype=torch.float32, non_blocking=True)
            n = h.shape[0]
            # Hinge discriminator update.
            for p in d.parameters():
                p.requires_grad_(True)
            opt_d.zero_grad(set_to_none=True)
            with torch.no_grad():
                h_fake = g(torch.randn(n, Z_DIM, device=device), c)
            loss_d = (F.relu(1.0 - d(h, c)).mean() + F.relu(1.0 + d(h_fake, c)).mean())
            loss_d.backward()
            opt_d.step()
            # Generator update: adversarial + differentiable physics loss.
            for p in d.parameters():
                p.requires_grad_(False)
            opt_g.zero_grad(set_to_none=True)
            fake = g(torch.randn(n, Z_DIM, device=device), c)
            loss_phys = physical_loss(fake, c)
            loss_g = -d(fake, c).mean() + args.lambda_physics * loss_phys
            loss_g.backward()
            opt_g.step()
            running += [loss_d.item(), loss_g.item(), loss_phys.item()]
            batches += 1
            if args.max_steps and batches >= args.max_steps:
                break
        means = running / max(1, batches)
        print(f'epoch {epoch:03d}/{args.epochs:03d} D={means[0]:.4f} '
              f'G={means[1]:.4f} physics={means[2]:.4f}', flush=True)
        with history_file.open('a', newline='') as f:
            csv.writer(f).writerow([epoch, *[f'{x:.6f}' for x in means]])
        if (epoch % args.save_every == 0) or epoch == args.epochs:
            ckpt = {'epoch': epoch, 'z_dim': Z_DIM, 'generator': g.state_dict(),
                    'discriminator': d.state_dict(), 'optim_g': opt_g.state_dict(),
                    'optim_d': opt_d.state_dict(), 'seed': args.seed,
                    'lambda_physics': args.lambda_physics}
            torch.save(ckpt, out / 'last.pt')
            if args.keep_epochs:
                torch.save(ckpt, out / f'epoch_{epoch:03d}.pt')
    print(f'Checkpoint: {out / "last.pt"}', flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', default='data/surfaces.npz')
    p.add_argument('--out', default='runs/microsurface')
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--batch', type=int, default=32)
    p.add_argument('--lr', type=float, default=0.0002)
    p.add_argument('--lambda-physics', type=float, default=5.0)
    p.add_argument('--seed', type=int, default=43)
    p.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--num-train', type=int, default=2400)
    p.add_argument('--num-val', type=int, default=300)
    p.add_argument('--save-every', type=int, default=5)
    p.add_argument('--keep-epochs', action='store_true')
    p.add_argument('--resume', default=None)
    p.add_argument('--max-steps', type=int, default=0,
                   help='Debug only: limit training batches per epoch.')
    args = p.parse_args()
    if args.batch < 1 or args.epochs < 1 or args.save_every < 1:
        p.error('batch, epochs, save-every must be positive')
    train(args)

if __name__ == '__main__':
    main()
