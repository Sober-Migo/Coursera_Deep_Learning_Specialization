## old architechture

# music_inference_model_out = [['InputLayer', [(None, 1, 90)], 0], ['InputLayer', [(None, 64)], 0], ['InputLayer', [(None, 64)], 0], ['LSTM', [(None, 64), (None, 64), (None, 64)], 39680, [(None, 1, 90), (None, 64), (None, 64)], 'tanh'], ['Dense', (None, 90), 5850, 'softmax'], ['TensorFlowOpLayer', [(None,)], 0], ['TensorFlowOpLayer', [(None, 90)], 0], ['RepeatVector', (None, 1, 90), 0, 1], ['TensorFlowOpLayer', [(None,)], 0], ['TensorFlowOpLayer', [(None, 90)], 0], ['RepeatVector', (None, 1, 90), 0, 1], ['TensorFlowOpLayer', [(None,)], 0], ['TensorFlowOpLayer', [(None, 90)], 0], ['RepeatVector', (None, 1, 90), 0, 1], ['TensorFlowOpLayer', [(None,)], 0], ['TensorFlowOpLayer', [(None, 90)], 0], ['RepeatVector', (None, 1, 90), 0, 1], ['TensorFlowOpLayer', [(None,)], 0], ['TensorFlowOpLayer', [(None, 90)], 0], ['RepeatVector', (None, 1, 90), 0, 1], ['TensorFlowOpLayer', [(None,)], 0], ['TensorFlowOpLayer', [(None, 90)], 0], ['RepeatVector', (None, 1, 90), 0, 1], ['TensorFlowOpLayer', [(None,)], 0], ['TensorFlowOpLayer', [(None, 90)], 0], ['RepeatVector', (None, 1, 90), 0, 1], ['TensorFlowOpLayer', [(None,)], 0], ['TensorFlowOpLayer', [(None, 90)], 0], ['RepeatVector', (None, 1, 90), 0, 1], ['TensorFlowOpLayer', [(None,)], 0], ['TensorFlowOpLayer', [(None, 90)], 0], ['RepeatVector', (None, 1, 90), 0, 1], ['TensorFlowOpLayer', [(None,)], 0], ['TensorFlowOpLayer', [(None, 90)], 0], ['RepeatVector', (None, 1, 90), 0, 1], ['TensorFlowOpLayer', [(None,)], 0], ['TensorFlowOpLayer', [(None, 90)], 0], ['RepeatVector', (None, 1, 90), 0, 1], ['TensorFlowOpLayer', [(None,)], 0], ['TensorFlowOpLayer', [(None, 90)], 0], ['RepeatVector', (None, 1, 90), 0, 1], ['TensorFlowOpLayer', [(None,)], 0], ['TensorFlowOpLayer', [(None, 90)], 0], ['RepeatVector', (None, 1, 90), 0, 1], ['TensorFlowOpLayer', [(None,)], 0], ['TensorFlowOpLayer', [(None, 90)], 0], ['RepeatVector', (None, 1, 90), 0, 1], ['TensorFlowOpLayer', [(None,)], 0], ['TensorFlowOpLayer', [(None, 90)], 0], ['RepeatVector', (None, 1, 90), 0, 1], ['TensorFlowOpLayer', [(None,)], 0], ['TensorFlowOpLayer', [(None, 90)], 0], ['RepeatVector', (None, 1, 90), 0, 1], ['TensorFlowOpLayer', [(None,)], 0], ['TensorFlowOpLayer', [(None, 90)], 0], ['RepeatVector', (None, 1, 90), 0, 1], ['TensorFlowOpLayer', [(None,)], 0], ['TensorFlowOpLayer', [(None, 90)], 0], ['RepeatVector', (None, 1, 90), 0, 1], ['TensorFlowOpLayer', [(None,)], 0], ['TensorFlowOpLayer', [(None, 90)], 0], ['RepeatVector', (None, 1, 90), 0, 1], ['TensorFlowOpLayer', [(None,)], 0], ['TensorFlowOpLayer', [(None, 90)], 0], ['RepeatVector', (None, 1, 90), 0, 1], ['TensorFlowOpLayer', [(None,)], 0], ['TensorFlowOpLayer', [(None, 90)], 0], ['RepeatVector', (None, 1, 90), 0, 1], ['TensorFlowOpLayer', [(None,)], 0], ['TensorFlowOpLayer', [(None, 90)], 0], ['RepeatVector', (None, 1, 90), 0, 1], ['TensorFlowOpLayer', [(None,)], 0], ['TensorFlowOpLayer', [(None, 90)], 0], ['RepeatVector', (None, 1, 90), 0, 1], ['TensorFlowOpLayer', [(None,)], 0], ['TensorFlowOpLayer', [(None, 90)], 0], ['RepeatVector', (None, 1, 90), 0, 1], ['TensorFlowOpLayer', [(None,)], 0], ['TensorFlowOpLayer', [(None, 90)], 0], ['RepeatVector', (None, 1, 90), 0, 1], ['TensorFlowOpLayer', [(None,)], 0], ['TensorFlowOpLayer', [(None, 90)], 0], ['RepeatVector', (None, 1, 90), 0, 1], ['TensorFlowOpLayer', [(None,)], 0], ['TensorFlowOpLayer', [(None, 90)], 0], ['RepeatVector', (None, 1, 90), 0, 1], ['TensorFlowOpLayer', [(None,)], 0], ['TensorFlowOpLayer', [(None, 90)], 0], ['RepeatVector', (None, 1, 90), 0, 1], ['TensorFlowOpLayer', [(None,)], 0], ['TensorFlowOpLayer', [(None, 90)], 0], ['RepeatVector', (None, 1, 90), 0, 1], ['TensorFlowOpLayer', [(None,)], 0], ['TensorFlowOpLayer', [(None, 90)], 0], ['RepeatVector', (None, 1, 90), 0, 1], ['TensorFlowOpLayer', [(None,)], 0], ['TensorFlowOpLayer', [(None, 90)], 0], ['RepeatVector', (None, 1, 90), 0, 1], ['TensorFlowOpLayer', [(None,)], 0], ['TensorFlowOpLayer', [(None, 90)], 0], ['RepeatVector', (None, 1, 90), 0, 1], ['TensorFlowOpLayer', [(None,)], 0], ['TensorFlowOpLayer', [(None, 90)], 0], ['RepeatVector', (None, 1, 90), 0, 1], ['TensorFlowOpLayer', [(None,)], 0], ['TensorFlowOpLayer', [(None, 90)], 0], ['RepeatVector', (None, 1, 90), 0, 1], ['TensorFlowOpLayer', [(None,)], 0], ['TensorFlowOpLayer', [(None, 90)], 0], ['RepeatVector', (None, 1, 90), 0, 1], ['TensorFlowOpLayer', [(None,)], 0], ['TensorFlowOpLayer', [(None, 90)], 0], ['RepeatVector', (None, 1, 90), 0, 1], ['TensorFlowOpLayer', [(None,)], 0], ['TensorFlowOpLayer', [(None, 90)], 0], ['RepeatVector', (None, 1, 90), 0, 1], ['TensorFlowOpLayer', [(None,)], 0], ['TensorFlowOpLayer', [(None, 90)], 0], ['RepeatVector', (None, 1, 90), 0, 1], ['TensorFlowOpLayer', [(None,)], 0], ['TensorFlowOpLayer', [(None, 90)], 0], ['RepeatVector', (None, 1, 90), 0, 1], ['TensorFlowOpLayer', [(None,)], 0], ['TensorFlowOpLayer', [(None, 90)], 0], ['RepeatVector', (None, 1, 90), 0, 1], ['TensorFlowOpLayer', [(None,)], 0], ['TensorFlowOpLayer', [(None, 90)], 0], ['RepeatVector', (None, 1, 90), 0, 1], ['TensorFlowOpLayer', [(None,)], 0], ['TensorFlowOpLayer', [(None, 90)], 0], ['RepeatVector', (None, 1, 90), 0, 1], ['TensorFlowOpLayer', [(None,)], 0], ['TensorFlowOpLayer', [(None, 90)], 0], ['RepeatVector', (None, 1, 90), 0, 1], ['TensorFlowOpLayer', [(None,)], 0], ['TensorFlowOpLayer', [(None, 90)], 0], ['RepeatVector', (None, 1, 90), 0, 1], ['TensorFlowOpLayer', [(None,)], 0], ['TensorFlowOpLayer', [(None, 90)], 0], ['RepeatVector', (None, 1, 90), 0, 1], ['TensorFlowOpLayer', [(None,)], 0], ['TensorFlowOpLayer', [(None, 90)], 0], ['RepeatVector', (None, 1, 90), 0, 1], ['TensorFlowOpLayer', [(None,)], 0], ['TensorFlowOpLayer', [(None, 90)], 0], ['RepeatVector', (None, 1, 90), 0, 1], ['TensorFlowOpLayer', [(None,)], 0], ['TensorFlowOpLayer', [(None, 90)], 0], ['RepeatVector', (None, 1, 90), 0, 1], ['TensorFlowOpLayer', [(None,)], 0], ['TensorFlowOpLayer', [(None, 90)], 0], ['RepeatVector', (None, 1, 90), 0, 1]]

# djmodel_out = [['InputLayer', [(None, 30, 90)], 0], ['TensorFlowOpLayer', [(None, 90)], 0], ['Reshape', (None, 1, 90), 0], ['InputLayer', [(None, 64)], 0], ['InputLayer', [(None, 64)], 0], ['TensorFlowOpLayer', [(None, 90)], 0], ['LSTM', [(None, 64), (None, 64), (None, 64)], 39680, [(None, 1, 90), (None, 64), (None, 64)], 'tanh'], ['TensorFlowOpLayer', [(None, 90)], 0], ['TensorFlowOpLayer', [(None, 90)], 0], ['TensorFlowOpLayer', [(None, 90)], 0], ['TensorFlowOpLayer', [(None, 90)], 0], ['TensorFlowOpLayer', [(None, 90)], 0], ['TensorFlowOpLayer', [(None, 90)], 0], ['TensorFlowOpLayer', [(None, 90)], 0], ['TensorFlowOpLayer', [(None, 90)], 0], ['TensorFlowOpLayer', [(None, 90)], 0], ['TensorFlowOpLayer', [(None, 90)], 0], ['TensorFlowOpLayer', [(None, 90)], 0], ['TensorFlowOpLayer', [(None, 90)], 0], ['TensorFlowOpLayer', [(None, 90)], 0], ['TensorFlowOpLayer', [(None, 90)], 0], ['TensorFlowOpLayer', [(None, 90)], 0], ['TensorFlowOpLayer', [(None, 90)], 0], ['TensorFlowOpLayer', [(None, 90)], 0], ['TensorFlowOpLayer', [(None, 90)], 0], ['TensorFlowOpLayer', [(None, 90)], 0], ['TensorFlowOpLayer', [(None, 90)], 0], ['TensorFlowOpLayer', [(None, 90)], 0], ['TensorFlowOpLayer', [(None, 90)], 0], ['TensorFlowOpLayer', [(None, 90)], 0], ['TensorFlowOpLayer', [(None, 90)], 0], ['TensorFlowOpLayer', [(None, 90)], 0], ['TensorFlowOpLayer', [(None, 90)], 0], ['TensorFlowOpLayer', [(None, 90)], 0], ['TensorFlowOpLayer', [(None, 90)], 0], ['Dense', (None, 90), 5850, 'softmax']]


# Modernized reference configurations scaled for dynamic vocabulary length (86 units)
# Recomputed weights: LSTM = 38656 parameters, Dense = 5590 parameters.

# Modernized Keras 3 true architectural target for djmodel (86 vocabulary scale)
djmodel_out = [
    ['InputLayer', (None, 30, 86), 0],
    ['Reshape', (None, 1, 86), 0],
    ['InputLayer', (None, 64), 0],
    ['InputLayer', (None, 64), 0],
    ['LSTM', (None, 64), 38656, (None, 1, 86), 'tanh'],
    ['Dense', (None, 86), 5590, 'softmax']
]

music_inference_model_out = [['InputLayer', (None, 1, 86), 0],
 ['InputLayer', (None, 64), 0],
 ['InputLayer', (None, 64), 0],
 ['LSTM', (None, 64), 38656, (None, 1, 86), 'tanh'],
 ['Dense', (None, 86), 5590, 'softmax'],
 ['RepeatVector', (None, 1, 86), 0, 1],
 ['RepeatVector', (None, 1, 86), 0, 1],
 ['RepeatVector', (None, 1, 86), 0, 1],
 ['RepeatVector', (None, 1, 86), 0, 1],
 ['RepeatVector', (None, 1, 86), 0, 1],
 ['RepeatVector', (None, 1, 86), 0, 1],
 ['RepeatVector', (None, 1, 86), 0, 1],
 ['RepeatVector', (None, 1, 86), 0, 1],
 ['RepeatVector', (None, 1, 86), 0, 1],
 ['RepeatVector', (None, 1, 86), 0, 1],
 ['RepeatVector', (None, 1, 86), 0, 1],
 ['RepeatVector', (None, 1, 86), 0, 1],
 ['RepeatVector', (None, 1, 86), 0, 1],
 ['RepeatVector', (None, 1, 86), 0, 1],
 ['RepeatVector', (None, 1, 86), 0, 1],
 ['RepeatVector', (None, 1, 86), 0, 1],
 ['RepeatVector', (None, 1, 86), 0, 1],
 ['RepeatVector', (None, 1, 86), 0, 1],
 ['RepeatVector', (None, 1, 86), 0, 1],
 ['RepeatVector', (None, 1, 86), 0, 1],
 ['RepeatVector', (None, 1, 86), 0, 1],
 ['RepeatVector', (None, 1, 86), 0, 1],
 ['RepeatVector', (None, 1, 86), 0, 1],
 ['RepeatVector', (None, 1, 86), 0, 1],
 ['RepeatVector', (None, 1, 86), 0, 1],
 ['RepeatVector', (None, 1, 86), 0, 1],
 ['RepeatVector', (None, 1, 86), 0, 1],
 ['RepeatVector', (None, 1, 86), 0, 1],
 ['RepeatVector', (None, 1, 86), 0, 1],
 ['RepeatVector', (None, 1, 86), 0, 1],
 ['RepeatVector', (None, 1, 86), 0, 1],
 ['RepeatVector', (None, 1, 86), 0, 1],
 ['RepeatVector', (None, 1, 86), 0, 1],
 ['RepeatVector', (None, 1, 86), 0, 1],
 ['RepeatVector', (None, 1, 86), 0, 1],
 ['RepeatVector', (None, 1, 86), 0, 1],
 ['RepeatVector', (None, 1, 86), 0, 1],
 ['RepeatVector', (None, 1, 86), 0, 1],
 ['RepeatVector', (None, 1, 86), 0, 1],
 ['RepeatVector', (None, 1, 86), 0, 1],
 ['RepeatVector', (None, 1, 86), 0, 1],
 ['RepeatVector', (None, 1, 86), 0, 1],
 ['RepeatVector', (None, 1, 86), 0, 1],
 ['RepeatVector', (None, 1, 86), 0, 1],
 ['RepeatVector', (None, 1, 86), 0, 1],
 ['RepeatVector', (None, 1, 86), 0, 1],
 ['RepeatVector', (None, 1, 86), 0, 1],
 ['RepeatVector', (None, 1, 86), 0, 1],
 ['RepeatVector', (None, 1, 86), 0, 1]]