import torch

def check_set_gpu(override=None):
    """
    Automatically checks for the best available device (CUDA, MPS, or CPU)
    and sets the torch device accordingly.
    """
    if override is None:
        if torch.backends.mps.is_available():
            device = torch.device('mps')
            print("Using GPU: Apple Metal Performance Shaders (MPS)")
            return device
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
            return device
        device = torch.device('cpu')
        print("Using CPU")
        return device
    else:
        device = torch.device(override)
    return device
