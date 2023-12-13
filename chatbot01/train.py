# training.py
import numpy as np
from nltk import NltkUtils


class Training:
    @staticmethod
    def softmax(x):
        exp_values = np.exp(x - np.max(x, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        return probabilities

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def regularization_term(weights, lambda_reg):
        return 0.5 * lambda_reg * np.sum(weights ** 2)

    @staticmethod
    def train_softmax(X_train, y_train, model, learning_rate, lambda_reg):
        scores_softmax = np.dot(X_train, model.weights_softmax) + model.biases_softmax
        probabilities_softmax = Training.softmax(scores_softmax)

        correct_probabilities_softmax = probabilities_softmax[range(len(y_train)), y_train]
        loss_softmax = -np.log(correct_probabilities_softmax + 1e-15)
        loss_softmax = np.mean(loss_softmax)

        regularization_softmax = Training.regularization_term(model.weights_softmax, lambda_reg)
        loss_softmax += regularization_softmax

        mask_softmax = np.zeros_like(probabilities_softmax)
        mask_softmax[range(len(y_train)), y_train] = -1 / (correct_probabilities_softmax + 1e-15)
        mask_softmax /= len(y_train)

        grad_weights_softmax = np.dot(X_train.T, mask_softmax) + lambda_reg * model.weights_softmax
        grad_biases_softmax = np.sum(mask_softmax, axis=0, keepdims=True)

        model.weights_softmax -= learning_rate * grad_weights_softmax
        model.biases_softmax -= learning_rate * grad_biases_softmax

        return loss_softmax