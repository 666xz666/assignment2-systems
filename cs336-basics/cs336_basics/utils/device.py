import torch

def try_gpu() -> torch.device:
    """优先自动选择GPU，没有可用GPU则使用CPU"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")