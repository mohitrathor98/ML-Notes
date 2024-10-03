
####
# explore the sigmoid function (also known as the logistic function)
# explore logistic regression; which uses the sigmoid function


import numpy as np
%matplotlib widget
import matplotlib.pyplot as plt
from plt_one_addpt_onclick import plt_one_addpt_onclick
from lab_utils_common import draw_vthresh
plt.style.use('./deeplearning.mplstyle')


##################################
# SIGMOID FUNCTION IMPLEMENTATION
##################################

# In the case of logistic regression, z (the input to the sigmoid function), is the output of a linear regression model.

# In the case of a single example,  𝑧 is scalar.

# in the case of multiple examples,  𝑧may be a vector consisting of  𝑚 values, one for each example.
# The implementation of the sigmoid function should cover both of these potential input formats. Let's implement this in Python.

# NumPy has a function called exp(), which offers a convenient way to calculate the exponential (𝑒^𝑧) of all elements in the input array (z).

# Input is an array. 
input_array = np.array([1,2,3])
exp_array = np.exp(input_array)

print("Input to exp:", input_array)
print("Output of exp:", exp_array)

# Input is a single number
input_val = 1  
exp_val = np.exp(input_val) # Calculates e^z value

print("Input to exp:", input_val)
print("Output of exp:", exp_val)

# Input to exp: [1 2 3]
# Output of exp: [ 2.72  7.39 20.09]
# Input to exp: 1
# Output of exp: 2.718281828459045

### Sigmoid Function ####
def sigmoid(z):
    """
    Compute the sigmoid of z

    Args:
        z (ndarray): A scalar, numpy array of any size.

    Returns:
        g (ndarray): sigmoid(z), with the same shape as z
         
    """

    g = 1/(1+np.exp(-z))
   
    return g


# Generate an array of evenly spaced values between -10 and 10
z_tmp = np.arange(-10,11)

# Use the function implemented above to get the sigmoid values
y = sigmoid(z_tmp)

# Code for pretty printing the two arrays next to each other
np.set_printoptions(precision=3) 
print("Input (z), Output (sigmoid(z))")
print(np.c_[z_tmp, y])

# Input (z), Output (sigmoid(z))
# [[-1.000e+01  4.540e-05]
#  [-9.000e+00  1.234e-04]
#  [-8.000e+00  3.354e-04]
#  [-7.000e+00  9.111e-04]
#  [-6.000e+00  2.473e-03]
#  [-5.000e+00  6.693e-03]
#  [-4.000e+00  1.799e-02]
#  [-3.000e+00  4.743e-02]
#  [-2.000e+00  1.192e-01]
#  [-1.000e+00  2.689e-01]
#  [ 0.000e+00  5.000e-01]
#  [ 1.000e+00  7.311e-01]
#  [ 2.000e+00  8.808e-01]
#  [ 3.000e+00  9.526e-01]
#  [ 4.000e+00  9.820e-01]
#  [ 5.000e+00  9.933e-01]
#  [ 6.000e+00  9.975e-01]
#  [ 7.000e+00  9.991e-01]
#  [ 8.000e+00  9.997e-01]
#  [ 9.000e+00  9.999e-01]
#  [ 1.000e+01  1.000e+00]]

#################################################################################################################
# The values in the left column are z, and the values in the right column are sigmoid(z). 
# As you can see, the input values to the sigmoid range from -10 to 10, and the output values range from 0 to 1.
#################################################################################################################