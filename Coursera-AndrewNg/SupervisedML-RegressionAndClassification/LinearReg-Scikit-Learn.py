
# https://scikit-learn.org/stable/index.html

## GOAL
# Utilize scikit-learn to implement linear regression using Gradient Descent

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from lab_utils_multi import  load_house_data
from lab_utils_common import dlc
np.set_printoptions(precision=2)
plt.style.use('./deeplearning.mplstyle')

# Scikit-learn has a gradient descent regression model sklearn.linear_model.SGDRegressor.
# (https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDRegressor.html#examples-using-sklearn-linear-model-sgdregressor)

# Like your previous implementation of gradient descent, this model performs best with normalized inputs. 
# sklearn.preprocessing.StandardScaler will perform z-score normalization as in a previous lab. 
# Here it is referred to as 'standard score'.
# (https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html#sklearn.preprocessing.StandardScaler)


### LOAD THE DATA SET
X_train, y_train = load_house_data()
X_features = ['size(sqft)','bedrooms','floors','age']

# --------------------------------------------------------------------------------------------------------------

### Normalize the Training Data
scaler = StandardScaler()
X_norm = scaler.fit_transform(X_train)
print(f"Peak to Peak range by column in Raw        X:{np.ptp(X_train,axis=0)}")   
print(f"Peak to Peak range by column in Normalized X:{np.ptp(X_norm,axis=0)}")

# Peak to Peak range by column in Raw        X:[2.41e+03 4.00e+00 1.00e+00 9.50e+01]
# Peak to Peak range by column in Normalized X:[5.85 6.14 2.06 3.69]


# ------------------------------------------------------------------------------------------------------------

### CREATE AND FIT REGRESSION MODEL

sgdr = SGDRegressor(max_iter=1000)
sgdr.fit(X_norm, y_train)
print(sgdr)
print(f"number of iterations completed: {sgdr.n_iter_}, number of weight updates: {sgdr.t_}")
# SGDRegressor(alpha=0.0001, average=False, early_stopping=False, epsilon=0.1,
#              eta0=0.01, fit_intercept=True, l1_ratio=0.15,
#              learning_rate='invscaling', loss='squared_loss', max_iter=1000,
#              n_iter_no_change=5, penalty='l2', power_t=0.25, random_state=None,
#              shuffle=True, tol=0.001, validation_fraction=0.1, verbose=0,
#              warm_start=False)
# number of iterations completed: 139, number of weight updates: 13762.0

### VIEW THE PARAMETERS
b_norm = sgdr.intercept_
w_norm = sgdr.coef_
print(f"model parameters:                   w: {w_norm}, b:{b_norm}")
print( "model parameters from previous lab: w: [110.56 -21.27 -32.71 -37.97], b: 363.16")

# model parameters:                   w: [110.27 -21.15 -32.58 -38.01], b:[363.15]
# model parameters from previous lab: w: [110.56 -21.27 -32.71 -37.97], b: 363.16


# -----------------------------------------------------------------------------------------------------------------------
### MAKE PREDICTIONS

# make a prediction using sgdr.predict()
y_pred_sgd = sgdr.predict(X_norm) # ---> Predicting result using Scikit learn algorithm


# make a prediction using w,b. 
y_pred = np.dot(X_norm, w_norm) + b_norm  # ---> Predicting result using (w.X + b) formula


print(f"prediction using np.dot() and sgdr.predict match: {(y_pred == y_pred_sgd).all()}")

print(f"Prediction on training set:\n{y_pred[:4]}" )
print(f"Target values \n{y_train[:4]}")

# prediction using np.dot() and sgdr.predict match: True
# Prediction on training set:
# [295.2  485.85 389.51 492.01]
# Target values 
# [300.  509.8 394.  540. ]
# --------------------------------------------------------------------------------------------------------------------