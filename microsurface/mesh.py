"""Export a regular heightfield as a triangulated OBJ mesh without 3D dependencies."""
from pathlib import Path
import numpy as np


def export_obj(height, filename, xy_scale=1.0, z_scale=1.0):
    h = np.asarray(height, dtype=np.float64)
    ny, nx = h.shape
    path = Path(filename); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w') as f:
        f.write('# MicroSurface-3D triangular height-field mesh\n')
        for y in range(ny):
            for x in range(nx):
                f.write(f'v {x*xy_scale:.6f} {y*xy_scale:.6f} {h[y,x]*z_scale:.6f}\n')
        for y in range(ny - 1):
            for x in range(nx - 1):
                a = y*nx + x + 1; b = a+1; c = a+nx; d = c+1
                f.write(f'f {a} {b} {d}\n')
                f.write(f'f {a} {d} {c}\n')
    return path
