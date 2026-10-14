"""Plot target/generated slope-anisotropy fidelity from the corrected metrics CSV.

The expected CSV column order is:
    target_slope,target_anis,generated_slope,generated_anis

Legacy exports before the header fix had swapped middle column *names*.
Do not use their mislabeled names without correcting the header first.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--csv', default='results/samples/sample_metrics.csv')
    p.add_argument('--output', default='docs/figures/anisotropy_fidelity.png')
    args = p.parse_args()

    data = np.genfromtxt(args.csv, delimiter=',', names=True)
    required = ('target_slope', 'target_anis', 'generated_slope', 'generated_anis')
    if data.dtype.names != required:
        raise SystemExit(f'Unexpected CSV columns: {data.dtype.names!r}; expected {required!r}')
    target = np.atleast_1d(data['target_anis'])
    generated = np.atleast_1d(data['generated_anis'])
    if target.size == 0:
        raise SystemExit('Empty CSV')

    lo, hi = min(target.min(), generated.min()), max(target.max(), generated.max())
    fig, ax = plt.subplots(figsize=(7.2, 5.2), constrained_layout=True)
    ax.scatter(target, generated, s=28, alpha=0.65, label='Generated surfaces')
    ax.plot([lo, hi], [lo, hi], linestyle='--', label='Ideal control')
    ax.set(xlabel='Target slope-anisotropy ratio',
           ylabel='Generated slope-anisotropy ratio',
           title=f'Conditional anisotropy control (validation, N={target.size})')
    corr = np.corrcoef(target, generated)[0, 1] if target.size >= 2 else float('nan')
    ax.text(0.03, 0.97,
            f'MAE = {np.abs(target - generated).mean():.4f}\nPearson r = {corr:.3f}',
            transform=ax.transAxes, ha='left', va='top',
            bbox=dict(facecolor='white', alpha=0.85, edgecolor='none'))
    ax.legend(loc='lower right')
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170)
    plt.close(fig)
    print(f'Saved {out} | anisotropy MAE = {np.abs(target - generated).mean():.6f}')


if __name__ == '__main__':
    main()
