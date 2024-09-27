import numpy as np
import matplotlib.pyplot as plt

# Problem Statement¶
# You would like a model which can predict housing prices given the size of the house.
# Let's use the same two data points as before the previous lab- a house with 1000 square feet sold for $300,000 and a house with 2000 square feet sold for $500,000.

# Size (1000 sqft)	Price (1000s of dollars)
# 1	300
# 2	500

x_train = np.array([1.0, 2.0])           #(size in 1000 square feet)
y_train = np.array([300.0, 500.0])           #(price in 1000s of dollars)

# Computing Cost
# The term 'cost' in this assignment might be a little confusing since the data is housing cost.
# Here, cost is a measure how well our model is predicting the target price of the house. The term 'price' is used for housing data.

# The equation for cost with one variable is:
# 𝐽(𝑤,𝑏)=12𝑚∑𝑖=0𝑚−1(𝑓𝑤,𝑏(𝑥(𝑖))−𝑦(𝑖))2(1)

# where
# 𝑓𝑤,𝑏(𝑥(𝑖))=𝑤𝑥(𝑖)+𝑏(2)
# 𝑓𝑤,𝑏(𝑥(𝑖))is our prediction for example 𝑖 using parameters 𝑤,𝑏

# (𝑓𝑤,𝑏(𝑥(𝑖))−𝑦(𝑖))2
#  is the squared difference between the target value and the prediction.
# These differences are summed over all the 𝑚 examples and divided by 2m to produce the cost, 𝐽(𝑤,𝑏)

# Note, in lecture summation ranges are typically from 1 to m, while code will be from 0 to m-1.

# The code below calculates cost by looping over each example. In each loop:

# f_wb, a prediction is calculated
# the difference between the target and the prediction is calculated and squared.
# this is added to the total cost.
def compute_cost(x, y, w, b): 
    """
    Computes the cost function for linear regression.
    
    Args:
      x (ndarray (m,)): Data, m examples 
      y (ndarray (m,)): target values
      w,b (scalar)    : model parameters  
    
    Returns
        total_cost (float): The cost of using w,b as the parameters for linear regression
               to fit the data points in x and y
    """
    # number of training examples
    m = x.shape[0] 
    
    cost_sum = 0 
    for i in range(m): 
        f_wb = w * x[i] + b   
        cost = (f_wb - y[i]) ** 2  
        cost_sum = cost_sum + cost  
    total_cost = (1 / (2 * m)) * cost_sum  

    return total_cost


# Your goal is to find a model  𝑓𝑤,𝑏(𝑥)=𝑤𝑥+𝑏
#  , with parameters  𝑤,𝑏
#  , which will accurately predict house values given an input  𝑥
#  . The cost is a measure of how accurate the model is on the training data.

# The cost equation (1) above shows that if  𝑤 and  𝑏
#   can be selected such that the predictions  𝑓𝑤,𝑏(𝑥)
#   match the target data  𝑦
#  , the  (𝑓𝑤,𝑏(𝑥(𝑖))−𝑦(𝑖))2
#   term will be zero and the cost minimized. In this simple two point example, you can achieve this!