import torch

# Check if CUDA is available
if torch.cuda.is_available():
    print("CUDA is available!")
    
    # Test CUDA functionality by creating a tensor on the GPU
    x = torch.tensor([1.0, 2.0, 3.0]).cuda()  # Move tensor to GPU
    print(f"Tensor on GPU: {x}")
    
    # Verify that the tensor is on the GPU
    print(f"Device: {x.device}")
else:
    print("CUDA is not available.")
