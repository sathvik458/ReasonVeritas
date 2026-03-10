#This is to check weather CUDA with GPU is working fine in system or not 
#This is not used in pipeline for any pourposes

import torch

print("PyTorch version:", torch.__version__)

# Check CUDA availability
cuda_available = torch.cuda.is_available()
print("CUDA Available:", cuda_available)

if cuda_available:
    print("CUDA Version:", torch.version.cuda)
    print("GPU Name:", torch.cuda.get_device_name(0))
    print("Number of GPUs:", torch.cuda.device_count())

    print("Current GPU:", torch.cuda.current_device())

    # Memory info
    print("Allocated Memory:", torch.cuda.memory_allocated(0))
    print("Cached Memory:", torch.cuda.memory_reserved(0))

    # Simple GPU tensor test
    x = torch.rand(3,3).cuda()
    y = torch.rand(3,3).cuda()

    print("\nMatrix Multiplication Test (GPU)")
    print(torch.matmul(x,y))

else:
    print("CUDA not detected. Running on CPU.")