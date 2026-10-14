"""Genuine adaptive triangle-mesh refinement of a height field.

Select vertices where the existing piecewise-linear Delaunay reconstruction
has highest absolute interpolation error; compare with a regular-grid mesh.
This is 2.5D adaptive sampling (not a general-purpose 3D remesher).
"""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import Delaunay
from scipy.interpolate import LinearNDInterpolator


def _unique_points(points):
    return np.unique(np.asarray(points,dtype=np.int32),axis=0)


def mesh_from_xy(height,xy):
    ny,nx=height.shape
    xy=_unique_points(xy)
    if (np.any(xy[:,0]<0) or np.any(xy[:,0]>=nx) or
        np.any(xy[:,1]<0) or np.any(xy[:,1]>=ny)):
        raise ValueError('Vertex outside image')
    tri=Delaunay(xy)
    z=height[xy[:,1],xy[:,0]]
    vertices=np.column_stack((xy,z)).astype(np.float32)
    return vertices,tri.simplices.astype(np.int32),tri


def reconstruct(height,xy):
    vertices,faces,tri=mesh_from_xy(height,xy)
    interp=LinearNDInterpolator(tri,vertices[:,2],fill_value=np.nan)
    ny,nx=height.shape
    yy,xx=np.mgrid[0:ny,0:nx]
    rec=np.asarray(interp(xx,yy),dtype=np.float32)
    if not np.isfinite(rec).all():
        raise RuntimeError('Interpolated grid not covered by mesh: include all corners')
    return rec,vertices,faces


def grid_points(size, count_per_axis):
    yy=np.unique(np.round(np.linspace(0,size-1,count_per_axis)).astype(np.int32))
    xx=yy
    return _unique_points([(x,y) for y in yy for x in xx])


def adaptive_refine(height,budget=500,initial_spacing=8, batch=32):
    ny,nx=height.shape
    if ny != nx:
        raise ValueError('This demo requires a square grid')
    if budget < 4 or budget > nx*ny:
        raise ValueError('Budget must be between 4 and number of pixels')
    grid=list(range(0,nx,initial_spacing))
    if grid[-1] != nx-1: grid.append(nx-1)
    xy=grid_points(nx, len(grid))  # approximately regular grid, including edges
    if len(xy)>budget:
        xy=grid_points(nx,max(2,int(np.floor(np.sqrt(budget)))))
    history=[]
    for step in range(200):
        rec,vertices,faces=reconstruct(height,xy)
        err=np.abs(height-rec)
        rmse=float(np.sqrt(np.mean((height-rec)**2)))
        history.append((len(xy),rmse))
        if len(xy)>=budget:
            break
        flat=np.argsort(err.ravel())[::-1]
        occupied=set(map(tuple,xy.tolist()))
        new=[]
        for idx in flat:
            y,x=divmod(int(idx),nx)
            if (x,y) not in occupied:
                new.append((x,y));occupied.add((x,y))
                if len(new)>=min(batch,budget-len(xy)):
                    break
        if not new:break
        xy=_unique_points(np.concatenate((xy,np.asarray(new,dtype=np.int32)),axis=0))
    rec,vertices,faces=reconstruct(height,xy)
    return rec,vertices,faces,history


def write_irregular_obj(vertices,faces,out):
    out=Path(out);out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('w') as f:
        f.write('# Adaptive triangular heightfield\n')
        for x,y,z in vertices: f.write(f'v {x:.5f} {y:.5f} {z:.5f}\n')
        for a,b,c in faces: f.write(f'f {a+1} {b+1} {c+1}\n')


def run(source='runs/microsurface/evaluation/samples.npz',
        out='runs/microsurface/mesh_refinement',budget=500):
    with np.load(source) as npz:
        h=np.asarray(npz['heights'][0],dtype=np.float32)
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    adapt_h,adapt_v,adapt_f,history=adaptive_refine(h,budget)
    n=int(round(np.sqrt(len(adapt_v))))
    reg_xy=grid_points(h.shape[0],n)
    regular_h,regular_v,regular_f=reconstruct(h,reg_xy)
    adapt_rmse=float(np.sqrt(np.mean((adapt_h-h)**2)))
    regular_rmse=float(np.sqrt(np.mean((regular_h-h)**2)))
    write_irregular_obj(adapt_v,adapt_f,out/'adaptive.obj')
    write_irregular_obj(regular_v,regular_f,out/'regular.obj')
    result={'adaptive_vertices':len(adapt_v),'regular_vertices':len(regular_v),
            'adaptive_triangles':len(adapt_f),'regular_triangles':len(regular_f),
            'adaptive_height_rmse':adapt_rmse,'regular_height_rmse':regular_rmse,
            'adaptive_vertex_budget':budget,
            'note':'Adaptive 2.5D Delaunay refinement; not a general 3D mesh operator.'}
    (out/'metrics.json').write_text(json.dumps(result,indent=2))
    fig,ax=plt.subplots(2,3,figsize=(13,8),constrained_layout=True)
    for a,img,name in [(ax[0,0],h,'Generated height'),
                        (ax[0,1],regular_h,'Regular triangulation'),
                        (ax[0,2],adapt_h,'Adaptive triangulation'),
                        (ax[1,0],np.abs(h-regular_h),'Regular abs. error'),
                        (ax[1,1],np.abs(h-adapt_h),'Adaptive abs. error')]:
        a.imshow(img,cmap='viridis');a.set_title(name);a.axis('off')
    ax[1,2].plot([i for i,_ in history],[rm for _,rm in history],marker='o')
    ax[1,2].set(xlabel='Number of vertices',ylabel='Height RMSE',title='Adaptive refinement curve')
    fig.savefig(out/'mesh_refinement.png',dpi=150);plt.close(fig)
    print(json.dumps(result,indent=2))
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source',default='runs/microsurface/evaluation/samples.npz')
    p.add_argument('--out',default='runs/microsurface/mesh_refinement')
    p.add_argument('--budget',type=int,default=500)
    a=p.parse_args();run(a.source,a.out,a.budget)

if __name__=='__main__':main()
