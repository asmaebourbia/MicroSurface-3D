"""All lengths are in grid/pixel units. Surface z(x,y) is a height field.

NDF figures show histograms of slopes (a *proxy* for the orientation
population); they are NOT calibrated microfacet distributions / measured BRDFs.
"""
import numpy as np
import torch


def gradients_np(h):
    h = np.asarray(h)
    # Periodic central differences, consistent with periodic synthetic surfaces.
    dx = (np.roll(h, -1, axis=-1) - np.roll(h, 1, axis=-1)) / 2.0
    dy = (np.roll(h, -1, axis=-2) - np.roll(h, 1, axis=-2)) / 2.0
    return dx, dy


def gradients_torch(h):
    dx = (torch.roll(h, -1, -1) - torch.roll(h, 1, -1)) / 2.0
    dy = (torch.roll(h, -1, -2) - torch.roll(h, 1, -2)) / 2.0
    return dx, dy


def stats_np(h):
    dx, dy = gradients_np(h)
    sx = float(np.sqrt(np.mean(dx * dx) + 1e-12))
    sy = float(np.sqrt(np.mean(dy * dy) + 1e-12))
    return np.asarray([np.hypot(sx, sy), sx / max(sy, 1e-8)], dtype=np.float32)


def stats_torch(h):
    """Input (B,1,H,W); output (B,2): RMS slope, directional ratio."""
    dx, dy = gradients_torch(h)
    sx = (dx.square().mean(dim=(1, 2, 3)) + 1e-10).sqrt()
    sy = (dy.square().mean(dim=(1, 2, 3)) + 1e-10).sqrt()
    return torch.stack(((sx.square() + sy.square()).sqrt(), sx / sy), dim=1)


def normals_np(h):
    dx, dy = gradients_np(h)
    n = np.stack((-dx, -dy, np.ones_like(dx)), axis=-1)
    return n / np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-8)


def default_lights():
    # Lights always in the upper hemisphere; positive irradiance for low slopes.
    xy = [(0,0), (0.65,0), (-0.65,0), (0,0.65), (0,-0.65),
          (0.45,0.45), (-0.45,-0.45), (0.45,-0.45), (-0.45,0.45)]
    l = np.asarray([(x,y,1.0) for x,y in xy], dtype=np.float32)
    return l / np.linalg.norm(l, axis=1, keepdims=True)


def render_lambertian(h, lights=None, albedo=0.8, noise_std=0.0, rng=None):
    """Controlled, linear Lambertian simulation (no interreflections)."""
    if lights is None:
        lights = default_lights()
    normals = normals_np(h)
    im = albedo * np.maximum(np.einsum('hwc,kc->khw', normals, lights), 0.0)
    if noise_std:
        if rng is None:
            rng = np.random.default_rng(0)
        im = im + rng.normal(0, noise_std, im.shape)
    return np.clip(im, 0.0, 1.0).astype(np.float32)


def estimate_normals(images, lights):
    """Photometric stereo with known calibrated lights, fixed albedo, no shadows."""
    k, hh, ww = images.shape
    if lights.shape != (k, 3):
        raise ValueError('lights must be (number_of_images,3)')
    g = np.linalg.pinv(lights.astype(np.float64)) @ images.reshape(k, -1)
    g = g.reshape(3, hh, ww).transpose(1, 2, 0)
    return (g / np.maximum(np.linalg.norm(g, axis=-1, keepdims=True), 1e-9)).astype(np.float32)


def integrate_normals(normals):
    """Periodic Fourier least-squares height integration, mean height = zero.

    Exactly inverts the periodic central-difference derivative on identifiable
    Fourier modes. Nyquist / nullspace modes cannot be recovered.
    """
    ny, nx = normals.shape[:2]
    p = -normals[..., 0] / np.maximum(normals[..., 2], 1e-7)
    q = -normals[..., 1] / np.maximum(normals[..., 2], 1e-7)
    ky = 1j * np.sin(2 * np.pi * np.fft.fftfreq(ny))[:, None]
    kx = 1j * np.sin(2 * np.pi * np.fft.fftfreq(nx))[None, :]
    denom = np.abs(kx)**2 + np.abs(ky)**2
    numer = np.conj(kx) * np.fft.fft2(p) + np.conj(ky) * np.fft.fft2(q)
    ht = np.zeros((ny, nx), dtype=np.complex128)
    np.divide(numer, denom, out=ht, where=denom > 1e-12)
    h = np.fft.ifft2(ht).real
    return (h - h.mean()).astype(np.float32)


def normal_mae_deg(a, b):
    cos = np.clip(np.sum(a*b, axis=-1), -1.0, 1.0)
    return float(np.rad2deg(np.arccos(cos)).mean())


def hist_normals(h, bins=40, value_range=(-0.8, 0.8)):
    n = normals_np(h)
    hist, _, _ = np.histogram2d(n[..., 0].ravel(), n[..., 1].ravel(),
                                 bins=bins, range=[value_range, value_range])
    hist = hist.astype(np.float64) + 1e-8
    return hist / hist.sum()


def js_divergence(p, q):
    p, q = np.asarray(p, dtype=np.float64), np.asarray(q, dtype=np.float64)
    p = p / p.sum(); q = q / q.sum(); m = 0.5 * (p + q)
    return float(0.5 * np.sum(p * np.log(p/m)) + 0.5 * np.sum(q * np.log(q/m)))
