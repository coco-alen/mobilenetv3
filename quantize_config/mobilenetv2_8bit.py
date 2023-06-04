config = [{
    'quant_types': ['input', 'weight'],
    'quant_bits': {'input': 8, 'weight': 8},
    'op_types': ['Conv2d']
}, {
    'quant_types': ['output'],
    'quant_bits': {'output': 8},
    'op_types': ['ReLU']
}, {
    'quant_types': ['input', 'weight'],
    'quant_bits': {'input': 8, 'weight': 8},
    'op_types': ['Linear']
}]