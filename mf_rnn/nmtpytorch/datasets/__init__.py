# First the basic types
from .text import TextDataset
from .numpy_sequence import NumpySequenceDataset

# Second the selector function
def get_dataset(type_):
    return {
        'numpysequence': NumpySequenceDataset,
        'text': TextDataset,
    }[type_.lower()]

# Should always be at the end
from .multimodal import MultimodalDataset