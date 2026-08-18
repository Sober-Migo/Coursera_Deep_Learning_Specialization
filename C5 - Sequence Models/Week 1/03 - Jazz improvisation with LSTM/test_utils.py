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
    if len(learner) != len(instructor):
        raise AssertionError("Error in test. The lists contain a different number of elements") 
    for index, a in enumerate(instructor):
        b = learner[index]
        if tuple(a) != tuple(b):
            print(colored(f"Test failed at index {index}", attrs=['bold']),
                  "\n Expected value \n\n", colored(f"{b}", "green"), 
                  "\n\n does not match the input value: \n\n", 
                  colored(f"{a}", "red"))
            raise AssertionError("Error in test") 
    print(colored("All tests passed!", "green"))

# extracts the description of a given model
# def summary(model):
#     model.compile(optimizer='adam',
#                   loss='categorical_crossentropy',
#                   metrics=['accuracy'])
#     result = []
#     for layer in model.layers:
#         descriptors = [layer.__class__.__name__, layer.output.shape, layer.count_params()]
#         if (type(layer) == Conv2D):
#             descriptors.append(layer.padding)
#             descriptors.append(layer.activation.__name__)
#             descriptors.append(layer.kernel_initializer.__class__.__name__)
#         if (type(layer) == MaxPooling2D):
#             descriptors.append(layer.pool_size)
#             descriptors.append(layer.strides)
#             descriptors.append(layer.padding)
#         if (type(layer) == Dropout):
#             descriptors.append(layer.rate)
#         if (type(layer) == ZeroPadding2D):
#             descriptors.append(layer.padding)
#         if (type(layer) == Dense):
#             descriptors.append(layer.activation.__name__)
#         if (type(layer) == LSTM):
#             descriptors.append(layer.input_shape)
#             descriptors.append(layer.activation.__name__)
#         if (type(layer) == RepeatVector):
#             descriptors.append(layer.n)
#         result.append(descriptors)
#     return result

# Overwriting the broken summary function in the local RAM environment
def summary(model):
    """
    Patched summary function compliant with Keras 3 multi-output list layers.
    Safely unpacks layer.output if it encounters a list sequence topology.
    """
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    result = []
    for layer in model.layers:
        # Check if the layer output is a list of tensors (Multi-output parameter sharing)
        if isinstance(layer.output, list):
            # Take the shape of the first tensor slice as the representative dimension
            layer_output_shape = layer.output[0].shape
        else:
            layer_output_shape = layer.output.shape
            
        descriptors = [layer.__class__.__name__, layer_output_shape, layer.count_params()]
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
            descriptors.append(layer.input[0].shape)
            descriptors.append(layer.activation.__name__)
        if (type(layer) == RepeatVector):
            descriptors.append(layer.n)
        result.append(descriptors)
    return result