## GOALS
# Utilize the multiple variables routines developed in the previous lab
# run Gradient Descent on a data set with multiple features
# explore the impact of the learning rate alpha on gradient descent
# improve performance of gradient descent by feature scaling using z-score normalization


import numpy as np
import matplotlib.pyplot as plt
from lab_utils_multi import  load_house_data, run_gradient_descent 
from lab_utils_multi import  norm_plot, plt_equal_scale, plot_cost_i_w
from lab_utils_common import dlc
np.set_printoptions(precision=2)
plt.style.use('./deeplearning.mplstyle')


# As in the previous labs, you will use the motivating example of housing price prediction. 
# The training data set contains many examples with 4 features (size, bedrooms, floors and age)
# shown in the table below. 
# Note, in this lab, the Size feature is in sqft while earlier labs utilized 1000 sqft.
# This data set is larger than the previous lab.


# load the dataset
X_train, y_train = load_house_data()
X_features = ['size(sqft)','bedrooms','floors','age']
print(f"X_train: {X_train}")
print(f"Y_train: {y_train}")

# X_train: [[1.24e+03 3.00e+00 1.00e+00 6.40e+01]
#  [1.95e+03 3.00e+00 2.00e+00 1.70e+01]
#  . . .
#  [1.05e+03 2.00e+00 1.00e+00 6.50e+01]]
# Y_train: [300.   509.8  394.  ....  257.8 ]

# ------------------------------
# Z-Score Normalization method 
#--------------------------------
def zscore_normalize_features(X):
    """
    computes  X, zcore normalized by column
    
    Args:
      X (ndarray (m,n))     : input data, m examples, n features
      
    Returns:
      X_norm (ndarray (m,n)): input normalized by column
      mu (ndarray (n,))     : mean of each feature
      sigma (ndarray (n,))  : standard deviation of each feature
    """
    # find the mean of each column/feature
    mu     = np.mean(X, axis=0)                 # mu will have shape (n,)
    # find the standard deviation of each column/feature
    sigma  = np.std(X, axis=0)                  # sigma will have shape (n,)
    # element-wise, subtract mu for that column from each example, divide by std for that column
    X_norm = (X - mu) / sigma      

    return (X_norm, mu, sig~ma)

#----------------------------------
    

# normalize the original features
X_norm, X_mu, X_sigma = zscore_normalize_features(X_train)
print(f"X_mu = {X_mu}, \nX_sigma = {X_sigma}")
print(f"Peak to Peak range by column in Raw        X:{np.ptp(X_train,axis=0)}")   
print(f"Peak to Peak range by column in Normalized X:{np.ptp(X_norm,axis=0)}")

# X_mu = [1.42e+03 2.72e+00 1.38e+00 3.84e+01], 
# X_sigma = [411.62   0.65   0.49  25.78]
# Peak to Peak range by column in Raw        X:[2.41e+03 4.00e+00 1.00e+00 9.50e+01]
# Peak to Peak range by column in Normalized X:[5.85 6.14 2.06 3.69]