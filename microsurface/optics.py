"""Anisotropic GGX *surrogate* for optical interpretation.

Important: mapping measured slope moments to GGX alpha parameters is a
heuristic and NOT a physically validated forward model of actual surfaces.
No Fresnel spectral measurement, multiple scattering, masking from geometry,
or experimental BRDF calibration is performed.
"""
import numpy as np


def slope_to_alpha(roughness, anisotropy):
    ratio = max(float(anisotropy), 1e-5)
    sx = roughness * ratio / np.sqrt(1 + ratio*ratio)
    sy = roughness / np.sqrt(1 + ratio*ratio)
    return max(0.02, np.sqrt(2)*sx), max(0.02, np.sqrt(2)*sy)


def _smith_g1(w, ax, ay):
    wz = max(float(w[2]), 1e-8)
    a2 = ((ax*w[0])**2 + (ay*w[1])**2) / (wz*wz)
    return 2.0 / (1.0 + np.sqrt(1.0 + a2))


def ggx_specular(view, light, alpha_x, alpha_y, f0=0.04):
    """Anisotropic GGX specular BRDF evaluation at one local direction pair."""
    v = np.asarray(view, dtype=np.float64); l = np.asarray(light, dtype=np.float64)
    v = v/np.linalg.norm(v); l = l/np.linalg.norm(l)
    if min(v[2], l[2]) <= 0:
        return 0.0
    halfway = v + l
    if np.linalg.norm(halfway) < 1e-10:
        return 0.0
    h = halfway / np.linalg.norm(halfway)
    if h[2] <= 0:
        return 0.0
    denom = ((h[0]/alpha_x)**2 + (h[1]/alpha_y)**2 + h[2]**2)**2
    D = 1.0/(np.pi*alpha_x*alpha_y*denom)
    F = f0 + (1-f0)*(1-max(float(np.dot(v, h)), 0.0))**5
    G = _smith_g1(v, alpha_x, alpha_y) * _smith_g1(l, alpha_x, alpha_y)
    return float(D*F*G/(4*v[2]*l[2]))


def ggx_curve(roughness, anisotropy, azimuth_deg=0, angles_deg=None):
    if angles_deg is None:
        angles_deg = np.linspace(0, 80, 81)
    az = np.deg2rad(azimuth_deg)
    ax, ay = slope_to_alpha(roughness, anisotropy)
    v = np.array([0., 0., 1.])
    values = [ggx_specular(v, [np.sin(np.deg2rad(t))*np.cos(az),
                                np.sin(np.deg2rad(t))*np.sin(az),
                                np.cos(np.deg2rad(t))], ax, ay)
              for t in angles_deg]
    return np.asarray(angles_deg), np.asarray(values)
