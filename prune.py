import torch
from params import args

from nni.compression.pytorch.pruning import L1NormPruner
from nni.compression.pytorch.speedup import ModelSpeedup

def prune_model(model:torch.nn.Module, config:list, dataLoader:torch.utils.data.DataLoader ):
    data, target = iter(dataLoader).next()
    pruner = L1NormPruner(model, config)
    print(model)
    # compress the model and generate the masks
    _, masks = pruner.compress()
    # show the masks sparsity
    for name, mask in masks.items():
        print(name, ' sparsity : ', '{:.2}'.format(mask['weight'].sum() / mask['weight'].numel()))
    pruner._unwrap_model()
    ModelSpeedup(model, data, masks).speedup_model()
    return model