import hashlib

def seed_hash(*args):
    """Derive an integer hash from all args, for use as a random seed."""
    
    args_str = str(args)
    return int(hashlib.md5(args_str.encode("utf-8")).hexdigest(), 16) % (2**31)

def set_seed(seed=42):
    """ Set random seed for all libraries. """
    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    