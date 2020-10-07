from .conditional import ConditionalDecoder
from .mf_decoder import MFDecoder


def get_decoder(type_):
    """Only expose ones with compatible __init__() arguments for now."""
    return {
        'cond': ConditionalDecoder,
        'mf': MFDecoder,
    }[type_]
