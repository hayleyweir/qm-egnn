import torch


from torch_geometric.transforms import ToDevice


def setup_cuda(
    gpu_index,
):  # Not needed for running on command line if variables are already set
    if not torch.cuda.is_available():
        raise ValueError("Cuda is not available")
    print(f"GPU is available: {torch.cuda.is_available()}")
    print(f"Number of GPUs available: {torch.cuda.device_count()}")
    device = torch.device(f"cuda:{gpu_index}")
    print(f"GPU name: {torch.cuda.get_device_name(device)}")
    print(f"GPU index: {device.index}")
    return device


class BatchToDevice:
    def __init__(self, device, keys=["edge_index", "pos", "z", "batch", "y"]):
        self.fn = ToDevice(device, keys)

    def __call__(self, data):
        return self.fn(data)
