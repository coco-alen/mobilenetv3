config = [{
    'sparsity_per_layer': 0.9,
    'op_types': ['Linear', 'Conv2d']
}, {
    'exclude': True,
    'op_names': ['classifier']
}]