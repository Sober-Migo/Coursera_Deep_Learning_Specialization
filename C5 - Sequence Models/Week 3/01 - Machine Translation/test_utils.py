from termcolor import colored

from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dropout 
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import ZeroPadding2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import RepeatVector




# Compare the two inputs
def comparator(learner, instructor):
    layer = 0
    for a, b in zip(learner, instructor):
        if tuple(a) != tuple(b):
            print(colored("Test failed", attrs=['bold']),
                  f"at layer: {layer}",
                  "\n Expected value \n\n", colored(f"{b}", "green"), 
                  "\n\n does not match the input value: \n\n", 
                  colored(f"{a}", "red"))
            raise AssertionError("Error in test") 
        layer += 1
    print(colored("All tests passed!", "green"))

# extracts the description of a given model
# extracts the description of a given model
def summary(model):
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    result = []
    
    # --- MAGIC HELPER FOR KERAS 3 COMPATIBILITY ---
    def get_keras_shape(obj, layer_name):
        # 1. Extract raw shapes from tensors (handles Keras 3 lists/nodes safely)
        if isinstance(obj, list):
            if isinstance(obj[0], list) or isinstance(obj[0], tuple): 
                # Shared layer with multiple tensors (e.g., LSTM returning state)
                shapes = [tuple(t.shape) for t in obj[0]]
            else:
                # Normal list of tensors (e.g., Dense in a loop, or multiple inputs)
                shapes = [tuple(t.shape) for t in obj if hasattr(t, 'shape')]
        else:
            # Single isolated tensor
            shapes = [tuple(obj.shape)]
            
        # 2. Format to match Coursera's exact Keras 2 expectations
        if layer_name == 'InputLayer':
            return shapes # Grader expects a list containing a tuple: [(None, ...)]
        elif layer_name == 'LSTM':
            return shapes # Grader expects a list of 3 tuples: [(None, ...), (None, ...), (None, ...)]
        else:
            return shapes[0] # Grader expects just a single tuple: (None, ...)
    # ----------------------------------------------
    
    for layer in model.layers:
        layer_name = layer.__class__.__name__
        
        # Safely get the output shape without triggering AttributeError
        out_shape = get_keras_shape(layer.output, layer_name)
        
        descriptors = [layer_name, out_shape, layer.count_params()]
        
        if (type(layer) == Conv2D):
            descriptors.append(layer.padding)
            descriptors.append(layer.activation.__name__)
            descriptors.append(layer.kernel_initializer.__class__.__name__)
        if (type(layer) == MaxPooling2D):
            descriptors.append(layer.pool_size)
            descriptors.append(layer.strides)
            descriptors.append(layer.padding)
        if (type(layer) == Dropout):
            descriptors.append(layer.rate)
        if (type(layer) == ZeroPadding2D):
            descriptors.append(layer.padding)
        if (type(layer) == Dense):
            descriptors.append(layer.activation.__name__)
        if (type(layer) == LSTM):
            # THE SECOND TRAP AVOIDED: Safely get the input shape
            in_shape = get_keras_shape(layer.input, layer_name)
            descriptors.append(in_shape)
            descriptors.append(layer.activation.__name__)
        if (type(layer) == RepeatVector):
            descriptors.append(layer.n)
            
        result.append(descriptors)
        
    return result