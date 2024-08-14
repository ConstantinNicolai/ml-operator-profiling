import sys
import importlib


if len(sys.argv) != 3:
    print("Usage: python script.py <model_name> <weights_file>")
    sys.exit(1)

model_name = sys.argv[1]
weights_file = sys.argv[2]

try:
    # Dynamically import the model module
    model_module = importlib.import_module(model_name)
    
    # Assuming the model class is named `Model` and is defined within the module
    # Adjust this if your model class has a different name or if you need to handle multiple classes
    model_class = getattr(model_module, 'Model')  
    
    # Instantiate the model
    model = model_class()
    
    # Load weights
    model.load_weights(weights_file)
    
    print(f"Successfully loaded model '{model_name}' with weights from '{weights_file}'")

except ModuleNotFoundError:
    print(f"Error: Module '{model_name}' not found")
except AttributeError:
    print(f"Error: The module '{model_name}' does not contain a 'Model' class")
except Exception as e:
    print(f"An error occurred: {e}")