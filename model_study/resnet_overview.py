import torch
import torch.nn as nn
from torchvision.models import resnet50
from torchvision import transforms
from torchsummary import summary

# Load the pretrained ResNet-50 model
model = resnet50(pretrained=True)

print(model)

# # ImageNet has 1000 classes, so we don't need to modify the model
# # However, if you want to modify it (e.g., for transfer learning), you could do something like this:
# # model.fc = nn.Linear(model.fc.in_features, num_classes)

# # Move the model to the appropriate device (GPU if available, otherwise CPU)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)

# # ImageNet input size is (3, 224, 224)
# input_size = (3, 224, 224)

# # Define a hook to register and print the shape of activations
# def print_activation_shape(module, input, output):
#     print(f'{module.__class__.__name__:>20}: {output.shape}')

# # Register the hook to all layers
# for layer in model.children():
#     layer.register_forward_hook(print_activation_shape)

# # Create a dummy input tensor with the ImageNet input size
# dummy_input = torch.randn(1, *input_size).to(device)

# # Perform a forward pass to trigger the hooks
# model(dummy_input)

# # Alternatively, use torchsummary to get a summary
# summary(model, input_size=input_size)
