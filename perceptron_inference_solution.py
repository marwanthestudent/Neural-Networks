import numpy as np
import os
import sys



#b : bias weight, float
#w : weights of the perceptron, col vector
#activation : activations function name in string, string
#input_vector: col vector

def perceptron_inference(b, w, activation, input_vector):

    a = np.dot(w.T,input_vector) + b #input for the activation function
    if activation.lower() == "sigmoid":
        z = 1/(1 + np.exp(-a))

    if activation.lower() == "step":
        if a < 0:
            z = 0
        elif a >= 0:
            z = 1

    return a,z




























