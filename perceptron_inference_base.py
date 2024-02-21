import numpy as np
import os
import sys
#from nn_load import *
from perceptron_inference_solution import perceptron_inference

# I added a read_matrix function since the original was not found - marwan
def read_matrix(weights_file):

    input = np.loadtxt(weights_file, dtype='f',delimiter=',')
    matrix = input.reshape(-1, 1) #since it's one entry per line, I had to change the shap of the matrix.

    return matrix



# Specify parameters for a test case.
weights_file = "weights1.txt"
input_file = "input1_11.txt"

activation_string = "step"
#activation_string = "sigmoid"

# Read the weights of the perceptron.
weights = read_matrix(weights_file)
b = weights[0,0]
w = weights[1:, :]

# Read the input vector.
input_vector = read_matrix(input_file)

# The next line is where your function is called.
(a, z) = perceptron_inference(b, w, activation_string, input_vector)

# Print the results.
print("a = %.4f\nz = %.4f" % (a, z))
