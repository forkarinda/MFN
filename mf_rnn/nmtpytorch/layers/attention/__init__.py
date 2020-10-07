from .mlp import MLPAttention
from .hierarchical import HierarchicalAttention
from .uniform import UniformAttention
from .dot import DotAttention

def get_attention(type_):
    return {
        'mlp': MLPAttention,
        'hier': HierarchicalAttention,
        'uniform': UniformAttention,
        'dot': DotAttention,
    }[type_]
