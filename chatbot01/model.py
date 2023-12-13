# model.py
import numpy as np

class Model:
    def __init__(self, num_features, num_classes):
        self.weights_softmax = np.random.rand(num_features, num_classes)
        self.biases_softmax = np.zeros((1, num_classes))
        self.weights_logistic = np.random.rand(num_features)
        self.bias_logistic = 0
