import sys

import torch

print(f"torch: {torch.__version__}")
print(f"torch CUDA build: {torch.version.cuda}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU count: {torch.cuda.device_count()}")
    for index in range(torch.cuda.device_count()):
        print(f"GPU {index}: {torch.cuda.get_device_name(index)}")
else:
    print("CUDA is unavailable. Check the NVIDIA driver and install with: uv sync --extra gpu")
    sys.exit(1)