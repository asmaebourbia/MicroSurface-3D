"""GAN condition fidelity, normals statistics and illustrative geometry/optics."""
import argparse
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from .data import Surfaces
from .models import load_generator, Z_DIM
from .physics import stats_np, hist_normals, js_divergence, normals_np
from .optics import ggx_curve
from .mesh import export_obj


def _figures(out, real_h, fake_h, conditions):
    n_show = min(4, len(fake_h))
    fig, ax = plt.subplots(n_show, 3, figsize=(10, n_show*2.8), squeeze=False,
                           constrained_layout=True)
    for j in range(n_show):
        for k, (arr, lab) in enumerate([(real_h[j], 'Reference (unpaired)'),
                                        (fake_h[j], 'GAN sample'),
                                        ((normals_np(fake_h[j])+1)/2, 'GAN normals RGB')]):
            ax[j, k].imshow(arr, cmap='viridis' if arr.ndim == 2 else None)
            ax[j, k].set_title(lab); ax[j, k].axis('off')
        ax[j, 1].set_xlabel(f'RMS={conditions[j,0]:.3f}; anis={conditions[j,1]:.2f}')
    fig.savefig(out/'comparison.png', dpi=140)
    plt.close(fig)
    # A real 3D surface plot (not just shaded 2D pixels).
    fig = plt.figure(figsize=(7,5))
    a = fig.add_subplot(111, projection='3d')
    ny, nx = fake_h[0].shape
    y, x = np.mgrid[0:ny, 0:nx]
    a.plot_surface(x, y, fake_h[0], cmap='viridis', linewidth=0, rstride=2, cstride=2)
    a.set(xlabel='x (px)', ylabel='y (px)', zlabel='height (grid units)',
          title='Generated triangulable height field')
    fig.savefig(out/'surface_3d.png', dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(9,4), constrained_layout=True)
    for axis, h, lab in [(axes[0], real_h[0], 'Real synthetic normals'),
                         (axes[1], fake_h[0], 'Generated normals')]:
        axis.imshow(hist_normals(h), origin='lower', cmap='magma', extent=(-.8,.8,-.8,.8))
        axis.set(xlabel='normal x', ylabel='normal y', title=f'{lab} (proxy)')
    fig.savefig(out/'normal_orientation_histograms.png', dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    real_c = conditions[0]
    gen_c = stats_np(fake_h[0])
    for axis, az in zip(axes, [0,90]):
        angles, a = ggx_curve(*real_c, azimuth_deg=az)
        _, b = ggx_curve(*gen_c, azimuth_deg=az)
        axis.plot(angles, a, label='Target-descriptor surrogate')
        axis.plot(angles, b, label='Generated-descriptor surrogate', linestyle='--')
        axis.set(xlabel='Light zenith angle (deg)', ylabel='Specular BRDF (sr^-1)',
                 title=f'Anisotropic GGX estimate, azimuth {az}°')
        axis.legend(fontsize=8)
    fig.savefig(out/'ggx_surrogate.png', dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7,4), constrained_layout=True)
    ax.scatter(conditions[:,0], [stats_np(h)[0] for h in fake_h],
               s=12, alpha=.7, label='Generated')
    ax.plot([.05,.32],[.05,.32], ls='--', color='black', label='Perfect match')
    ax.set(xlabel='Target RMS slope', ylabel='Generated RMS slope',
           title='Conditional control (validation split)', xlim=(.05,.32), ylim=(.05,.32))
    ax.legend(); fig.savefig(out/'conditional_fidelity.png', dpi=150)
    plt.close(fig)


def evaluate(checkpoint='runs/microsurface/last.pt', data='data/surfaces.npz',
             out='runs/microsurface/evaluation', count=100, seed=44, device='auto'):
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    dev = 'cuda' if (device == 'auto' and torch.cuda.is_available()) else (
          'cpu' if device == 'auto' else device)
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    g = load_generator(checkpoint, dev)
    ds = Surfaces(data, 'val')
    n = min(count, len(ds))
    if n < 1:
        raise ValueError('No validation examples')
    # Same condition, unpaired surfaces! Pixel RMSE is deliberately NOT used.
    ref = ds.h[:n]
    cond = ds.c[:n]
    torch.manual_seed(seed)
    samples = []
    with torch.no_grad():
        for i in range(0,n,32):
            cc = torch.tensor(cond[i:i+32],device=dev)
            zz = torch.randn(len(cc), Z_DIM, device=dev)
            samples.append(g(zz, cc).cpu().numpy()[:,0])
    fake = np.concatenate(samples,0)
    measured = np.stack([stats_np(h) for h in fake])
    rough_mae = float(np.abs(measured[:,0]-cond[:,0]).mean())
    ratio_mae = float(np.abs(measured[:,1]-cond[:,1]).mean())
    log_ratio_mae = float(np.abs(np.log(measured[:,1])-np.log(cond[:,1])).mean())
    # Aggregate distribution proxy: compares normal orientations, not BRDF.
    ref_hist = np.mean(np.stack([hist_normals(h) for h in ref]),0)
    fake_hist = np.mean(np.stack([hist_normals(h) for h in fake]),0)
    # Diversity at fixed condition: pairwise height RMSE across independent z.
    fixed = torch.tensor(cond[:1], device=dev).repeat(8,1)
    with torch.no_grad():
        reps = g(torch.randn(8,Z_DIM,device=dev),fixed).cpu().numpy()[:,0]
    diversity = float(np.mean([np.sqrt(np.mean((reps[i]-reps[j])**2))
                               for i in range(8) for j in range(i+1,8)]))
    result = {'num_validation_samples':n, 'rms_slope_mae':rough_mae,
              'anisotropy_ratio_mae':ratio_mae, 'log_anisotropy_mae':log_ratio_mae,
              'normal_orientation_js_proxy':js_divergence(ref_hist,fake_hist),
              'fixed_condition_pairwise_height_rmse':diversity,
              'notes':['No paired geometry targets: many heights may share one optical descriptor.',
                       'Orientation histogram is a proxy, not a calibrated NDF.',
                       'GGX curves are a descriptor-based surrogate, not measured BRDF.']}
    (out/'metrics.json').write_text(json.dumps(result,indent=2,ensure_ascii=False))
    np.savez_compressed(out/'samples.npz', heights=fake, conditions=cond, reference=ref)
    export_obj(fake[0],out/'generated_surface.obj')
    with (out/'sample_metrics.csv').open('w',newline='') as f:
        writer=csv.writer(f)
        writer.writerow(['target_slope','target_anis','generated_slope','generated_anis'])
        for c,m in zip(cond,measured): writer.writerow([*c,*m])
    _figures(out,ref,fake,cond)
    print(json.dumps(result,indent=2,ensure_ascii=False))
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint', default='runs/microsurface/last.pt')
    p.add_argument('--data',default='data/surfaces.npz')
    p.add_argument('--out',default='runs/microsurface/evaluation')
    p.add_argument('--count',type=int,default=100)
    p.add_argument('--seed',type=int,default=44)
    p.add_argument('--device',choices=['cpu','cuda','auto'],default='auto')
    a=p.parse_args();evaluate(a.checkpoint,a.data,a.out,a.count,a.seed,a.device)

if __name__=='__main__':
    main()
