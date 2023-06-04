
import torch
import torch.quantization as quant
from nni.algorithms.compression.pytorch.quantization import QAT_Quantizer
from params import args

def quantize_model(model:torch.nn.Module, config:list, dataLoader:torch.utils.data.DataLoader, optimizer:torch.optim.Optimizer):
    data, target = iter(dataLoader).next()
    quantizer = QAT_Quantizer(model, config, optimizer, data)
    quantizer.compress()


