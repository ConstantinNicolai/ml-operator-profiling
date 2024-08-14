import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50
from torchvision import transforms
from torchsummary import summary
from torch_profiling_utils.fvcorewriter import FVCoreWriter
from torch_profiling_utils.torchinfowriter import TorchinfoWriter

# Load the pretrained ResNet-50 model
model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)




class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        
        # First Convolutional Layer
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        
        # Second Convolutional Layer
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        
        # Third Convolutional Layer
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        
        # Fully Connected Layer (Linear Layer)
        # Assuming input images are 32x32
        self.fc1 = nn.Linear(64 * 4 * 4, num_classes)

    def forward(self, x):
        # Pass through the first conv layer and apply ReLU activation
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)  # Apply max pooling
        
        # Pass through the second conv layer and apply ReLU activation
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)  # Apply max pooling
        
        # Pass through the third conv layer and apply ReLU activation
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)  # Apply max pooling
        
        # Flatten the output from the conv layers before passing into the fully connected layer
        x = x.view(x.size(0), -1)
        
        # Pass through the fully connected layer
        x = self.fc1(x)
        
        return x

#print(model)


# model = SimpleCNN(num_classes=10)
print(model)


# # ImageNet has 1000 classes, so we don't need to modify the model
# # However, if you want to modify it (e.g., for transfer learning), you could do something like this:
# # model.fc = nn.Linear(model.fc.in_features, num_classes)

# # Move the model to the appropriate device (GPU if available, otherwise CPU)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)

# ImageNet input size is (3, 224, 224)
input_size = (3, 32, 32)

# # Define a hook to register and print the shape of activations
# def print_activation_shape(module, input, output):
#     print(f'{module.__class__.__name__:>20}: {output.shape}')

# # Register the hook to all layers
# for layer in model.children():
#     layer.register_forward_hook(print_activation_shape)

# Create a dummy input tensor with the ImageNet input size
input_data = torch.randn(1, *input_size)
#dummy_input = torch.randn(1, *input_size).to(device)

# # Perform a forward pass to trigger the hooks
# model(dummy_input)

# # Alternatively, use torchsummary to get a summary
# summary(model, input_size=input_size)



fvcore_writer = FVCoreWriter(model, input_data)

fvcore_writer.get_flop_dict('by_module')
fvcore_writer.get_flop_dict('by_operator')

fvcore_writer.get_activation_dict('by_module')
fvcore_writer.get_activation_dict('by_operator')

fvcore_writer.write_flops_to_json("output_test.json", 'by_module')

fvcore_writer.write_activations_to_json("operator_fvcore.json",'by_operator')



torchinfo_writer = TorchinfoWriter(model,
                                    input_data=input_data,
                                    verbose=0)

torchinfo_writer.construct_model_tree()

torchinfo_writer.show_model_tree(attr_list=['Name', 'Kernel Size', 'Input Size', 'Output Size', 'Parameters', 'MACs'])