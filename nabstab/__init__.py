try:
    import torch
except ImportError:
    raise ImportError(
        "PyTorch is required but not installed. Please install it first:\n"
        "CPU: pip install torch --index-url https://download.pytorch.org/whl/cpu\n" 
        "GPU: pip install torch --index-url https://download.pytorch.org/whl/cu118"
    )

