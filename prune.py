import torch
from params import args

from nni.compression.pytorch.pruning import L1NormPruner

def prune_model(model:torch.nn.Module, config:list ):
    pruner = L1NormPruner(model, config)
    print(model)
    # compress the model and generate the masks
    _, masks = pruner.compress()
    # show the masks sparsity
    for name, mask in masks.items():
        print(name, ' sparsity : ', '{:.2}'.format(mask['weight'].sum() / mask['weight'].numel()))

    return model