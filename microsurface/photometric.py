"""Independent photometric stereo reconstruction experiment."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from .data import synthetic_surface
from .mesh import export_obj
from .physics import (default_lights, render_lambertian, estimate_normals,
                      integrate_normals, normals_np, normal_mae_deg)


def run(out='runs/microsurface/photometric', size=64, seed=43, noise=0.002):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    original = synthetic_surface(size, rng)
    lights = default_lights()
    imgs = render_lambertian(original, lights, noise_std=noise, rng=rng)
    normals_hat = estimate_normals(imgs, lights)
    heights_hat = integrate_normals(normals_hat)
    ref_normal = normals_np(original)
    metrics = {
        'height_rmse_grid_units': float(np.sqrt(np.mean((heights_hat - original)**2))),
        'normal_mean_angular_error_deg': normal_mae_deg(ref_normal, normals_hat),
        'num_lights': len(lights), 'noise_std': noise,
        'note': 'idealized Lambertian, calibrated lights, periodic surface, no occlusions'}
    (out/'metrics.json').write_text(json.dumps(metrics, indent=2))
    np.savez_compressed(out/'reconstruction.npz', original=original,
                        recovered=heights_hat, images=imgs, normals=normals_hat)
    export_obj(heights_hat, out/'reconstruction.obj')
    fig, axes = plt.subplots(2, 4, figsize=(12, 6), constrained_layout=True)
    for i in range(4):
        axes[0, i].imshow(imgs[i], cmap='gray', vmin=0, vmax=1)
        axes[0, i].set_title(f'Lighting {i+1}'); axes[0, i].axis('off')
    for ax, arr, name in zip(axes[1],
                            [original, heights_hat, np.abs(original-heights_hat),
                             (normals_hat+1)/2],
                            ['Reference height', 'Reconstructed height', 'Absolute error', 'Estimated normals']):
        ax.imshow(arr, cmap='viridis' if arr.ndim == 2 else None)
        ax.set_title(name); ax.axis('off')
    fig.savefig(out/'photometric_stereo.png', dpi=160)
    plt.close(fig)
    print(json.dumps(metrics, indent=2))
    return metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', default='runs/microsurface/photometric')
    p.add_argument('--size', type=int, default=64)
    p.add_argument('--seed', type=int, default=43)
    p.add_argument('--noise', type=float, default=0.002)
    a = p.parse_args(); run(a.out, a.size, a.seed, a.noise)

if __name__ == '__main__':
    main()
