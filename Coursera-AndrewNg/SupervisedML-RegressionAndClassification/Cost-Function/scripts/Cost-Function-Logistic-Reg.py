# examine the implementation and utilize the cost function for logistic regression

import numpy as np
%matplotlib widget
import matplotlib.pyplot as plt
from lab_utils_common import  plot_data, sigmoid, dlc
plt.style.use('./deeplearning.mplstyle')

## DATASET
X_train = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])  #(m,n)
y_train = np.array([0, 0, 0, 1, 1, 1])                                           #(m,)

# Plotting the data set
fig,ax = plt.subplots(1,1,figsize=(4,4))
plot_data(X_train, y_train, ax)

# Set both axes to be from 0-4
ax.axis([0, 4, 0, 3.5])
ax.set_ylabel('$x_1$', fontsize=12)
ax.set_xlabel('$x_0$', fontsize=12)
plt.show()
# In Pic: DataSet-Plotting-For-CostFunction.png

#############################################################################
## COMPUTING COST USING FORMULA OF COST FUNCTION FOR LOGISTIC REGRESSIONS
#############################################################################
# variables X and y are not scalar values but matrices of shape ( 𝑚,𝑛 ) and ( 𝑚,) respectively, 
# where  𝑛 is the number of features and  𝑚 is the number of training examples.

def compute_cost_logistic(X, y, w, b):
    """
    Computes cost

    Args:
      X (ndarray (m,n)): Data, m examples with n features
      y (ndarray (m,)) : target values
      w (ndarray (n,)) : model parameters  
      b (scalar)       : model parameter
      
    Returns:
      cost (scalar): cost
    """

    m = X.shape[0]
    cost = 0.0
    # Summation of cost for all the data set
    for i in range(m):
        # Z = W.X + b
        z_i = np.dot(X[i],w) + b
        # Predicted output
        f_wb_i = sigmoid(z_i)
        # Calculating cost using formula -- See lecture notes.
        cost +=  -y[i]*np.log(f_wb_i) - (1-y[i])*np.log(1-f_wb_i)
             
    cost = cost / m
    return cost


w_tmp = np.array([1,1])
b_tmp = -3
print(compute_cost_logistic(X_train, y_train, w_tmp, b_tmp))

# 0.36686678640551745