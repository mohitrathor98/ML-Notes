
## GOAL
# explore feature engineering and polynomial regression which allows you to use
# the machinery of linear regression to fit very complicated, even very non-linear functions.

import numpy as np
import matplotlib.pyplot as plt
from lab_utils_multi import zscore_normalize_features, run_gradient_descent_feng
np.set_printoptions(precision=2)  # reduced display precision on numpy arrays

#------------------------------------------------------------------------------------------------------------------
# Let's try using what we know so far to fit a non-linear curve. We'll start with a simple quadratic:  𝑦=1+𝑥2

# We'll use np.c_[..] which is a NumPy routine to concatenate along the column boundary.
# https://numpy.org/doc/stable/reference/generated/numpy.c_.html

# create target data
x = np.arange(0, 20, 1)
y = 1 + x**2
X = x.reshape(-1, 1)

model_w,model_b = run_gradient_descent_feng(X,y,iterations=1000, alpha = 1e-2)

plt.scatter(x, y, marker='x', c='r', label="Actual Value"); plt.title("no feature engineering")
plt.plot(x,X@model_w + model_b, label="Predicted Value");  plt.xlabel("X"); plt.ylabel("y"); plt.legend(); plt.show()


# Iteration         0, Cost: 1.65756e+03
# Iteration       100, Cost: 6.94549e+02
# Iteration       200, Cost: 5.88475e+02
# Iteration       300, Cost: 5.26414e+02
# Iteration       400, Cost: 4.90103e+02
# Iteration       500, Cost: 4.68858e+02
# Iteration       600, Cost: 4.56428e+02
# Iteration       700, Cost: 4.49155e+02
# Iteration       800, Cost: 4.44900e+02
# Iteration       900, Cost: 4.42411e+02
# w,b found by gradient descent: w: [18.7], b: -52.0834


# ---------------------------------------------------------------------------

# The above equation is not a great fit, if we plot a graph with it: Feature-Engineering-and-Polynomial-Graph1-png

# What is needed is something like  𝑦= w.x^2 + b, or a polynomial feature.

# modify the input data to engineer the needed features. If you swap the original data with a version that squares the 𝑥
# value, then you can achieve the required equation

# create target data
x = np.arange(0, 20, 1)
y = 1 + x**2

# Engineer features 
X = x**2      #<-- added engineered feature

X = X.reshape(-1, 1)  #X should be a 2-D Matrix
model_w,model_b = run_gradient_descent_feng(X, y, iterations=10000, alpha = 1e-5)

plt.scatter(x, y, marker='x', c='r', label="Actual Value"); plt.title("Added x**2 feature")
plt.plot(x, np.dot(X,model_w) + model_b, label="Predicted Value"); plt.xlabel("x"); plt.ylabel("y"); plt.legend(); plt.show()

# Iteration         0, Cost: 7.32922e+03
# Iteration      1000, Cost: 2.24844e-01
# Iteration      2000, Cost: 2.22795e-01
# Iteration      3000, Cost: 2.20764e-01
# Iteration      4000, Cost: 2.18752e-01
# Iteration      5000, Cost: 2.16758e-01
# Iteration      6000, Cost: 2.14782e-01
# Iteration      7000, Cost: 2.12824e-01
# Iteration      8000, Cost: 2.10884e-01
# Iteration      9000, Cost: 2.08962e-01
# w,b found by gradient descent: w: [1.], b: 0.0490

# Graph: Feature-Engineering-and-Polynomial-Graph2.png


# -------------------------------------------------------------------------------------

## SCALING FEATURE

# if the data set has features with significantly different scales,
# one should apply feature scaling to speed gradient descent.

# create target data
x = np.arange(0,20,1)
X = np.c_[x, x**2, x**3]
print(f"Peak to Peak range by column in Raw        X:{np.ptp(X,axis=0)}")

# add mean_normalization 
X = zscore_normalize_features(X)     
print(f"Peak to Peak range by column in Normalized X:{np.ptp(X,axis=0)}")

# Peak to Peak range by column in Raw        X:[  19  361 6859]
# Peak to Peak range by column in Normalized X:[3.3  3.18 3.28]

x = np.arange(0,20,1)
y = x**2

X = np.c_[x, x**2, x**3]
X = zscore_normalize_features(X) 

model_w, model_b = run_gradient_descent_feng(X, y, iterations=100000, alpha=1e-1)

plt.scatter(x, y, marker='x', c='r', label="Actual Value"); plt.title("Normalized x x**2, x**3 feature")
plt.plot(x,X@model_w + model_b, label="Predicted Value"); plt.xlabel("x"); plt.ylabel("y"); plt.legend(); plt.show()

# Iteration         0, Cost: 9.42147e+03
# Iteration     10000, Cost: 3.90938e-01
# Iteration     20000, Cost: 2.78389e-02
# Iteration     30000, Cost: 1.98242e-03
# Iteration     40000, Cost: 1.41169e-04
# Iteration     50000, Cost: 1.00527e-05
# Iteration     60000, Cost: 7.15855e-07
# Iteration     70000, Cost: 5.09763e-08
# Iteration     80000, Cost: 3.63004e-09
# Iteration     90000, Cost: 2.58497e-10
# w,b found by gradient descent: w: [5.27e-05 1.13e+02 8.43e-05], b: 123.5000

# GRAPH: Feature-Engineering-and-Polynomial-Graph3.png