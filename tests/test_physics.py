import numpy as np
import torch
from microsurface.physics import (stats_np,stats_torch,normals_np,default_lights,
                                  render_lambertian,estimate_normals,integrate_normals)
from microsurface.data import synthetic_surface
from microsurface.models import Generator,Discriminator,Z_DIM


def test_photometric_stereo_no_noise():
    h=synthetic_surface(64,np.random.default_rng(43))
    lights=default_lights(); imgs=render_lambertian(h,lights,noise_std=0)
    est=estimate_normals(imgs,lights)
    assert np.mean(np.abs(est-normals_np(h))) < 1e-5
    rec=integrate_normals(est)
    assert np.sqrt(np.mean((rec-h)**2)) < 0.12


def test_numpy_torch_conditions():
    h=synthetic_surface(64,np.random.default_rng(0))
    a=stats_np(h)
    b=stats_torch(torch.tensor(h).reshape(1,1,64,64)).numpy()[0]
    np.testing.assert_allclose(a,b,atol=1e-5)


def test_gan_shape():
    g=Generator();d=Discriminator()
    c=torch.tensor([[.2,1.0],[.15,.8]])
    h=g(torch.randn(2,Z_DIM),c)
    assert h.shape==(2,1,64,64)
    assert d(h,c).shape==(2,)
