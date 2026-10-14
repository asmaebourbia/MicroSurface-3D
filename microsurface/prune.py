"""Exploratory unstructured magnitude pruning (not a speedup claim)."""
import argparse
import copy
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn.utils import prune
from .models import load_generator, Z_DIM
from .physics import stats_np


def run(checkpoint, data, out, amount=.20, device='auto'):
    if not 0 <= amount < 1:
        raise ValueError('pruning amount must be in [0,1)')
    dev='cuda' if device=='auto' and torch.cuda.is_available() else ('cpu' if device=='auto' else device)
    g=load_generator(checkpoint,dev)
    pruned=copy.deepcopy(g)
    layers=[(m,'weight') for m in pruned.modules()
            if isinstance(m,(nn.Linear,nn.ConvTranspose2d,nn.Conv2d))]
    prune.global_unstructured(layers,prune.L1Unstructured,amount=amount)
    for m,name in layers: prune.remove(m,name)
    from .data import Surfaces
    c=Surfaces(data,'val').c[:32]
    torch.manual_seed(100)
    with torch.no_grad():
        cc=torch.tensor(c,device=dev)
        z=torch.randn(len(cc),Z_DIM,device=dev)
        a=g(z,cc).cpu().numpy()[:,0]
        b=pruned(z,cc).cpu().numpy()[:,0]
    errors1=np.mean(np.abs(np.stack([stats_np(h) for h in a])-c),axis=0)
    errors2=np.mean(np.abs(np.stack([stats_np(h) for h in b])-c),axis=0)
    weights=[m.weight.detach() for m,_ in layers]
    total=sum(x.numel() for x in weights)
    sparse=sum(int((x==0).sum()) for x in weights)
    result={'target_pruning_fraction':amount,'actual_weight_sparsity':sparse/total,
            'pre_pruning_slope_mae':float(errors1[0]),'post_pruning_slope_mae':float(errors2[0]),
            'pre_pruning_anisotropy_mae':float(errors1[1]),
            'post_pruning_anisotropy_mae':float(errors2[1]),
            'mean_height_difference':float(np.mean(np.abs(a-b))),
            'note':'Unstructured zero weights DO NOT imply lower dense GPU runtime or smaller state_dict.'}
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    torch.save({'generator':pruned.state_dict(),'z_dim':Z_DIM,'pruned_fraction':amount},out/'generator_pruned.pt')
    (out/'metrics.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',default='runs/microsurface/last.pt')
    p.add_argument('--data',default='data/surfaces.npz')
    p.add_argument('--out',default='runs/microsurface/pruning')
    p.add_argument('--amount',type=float,default=.20)
    p.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    a=p.parse_args();run(a.checkpoint,a.data,a.out,a.amount,a.device)

if __name__=='__main__':main()
