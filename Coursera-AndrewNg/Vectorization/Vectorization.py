import numpy as np    # it is an unofficial standard to use np for numpy
import time

# NumPy Documentation including a basic introduction: NumPy.org
# A challenging feature topic: https://numpy.org/doc/stable/user/basics.broadcasting.html

# 3 Vectors¶

# 3.1 Abstract
# Vectors, as you will use them in this course, are ordered arrays of numbers. In notation, vectors are denoted with lower case bold letters such as 𝐱
#  . The elements of a vector are all the same type. A vector does not, for example, contain both characters and numbers. 
#   The number of elements in the array is often referred to as the dimension though mathematicians 
# may prefer rank. The vector shown has a dimension of  𝑛
#  . The elements of a vector can be referenced with an index. In math settings, indexes typically run from 1 to n. 


# 3.2 NumPy Arrays
# NumPy's basic data structure is an indexable, n-dimensional array containing elements of the same type (dtype).
#  Right away, you may notice we have overloaded the term 'dimension'. Above, it was the number of elements in the vector,
# here, dimension refers to the number of indexes of an array. A one-dimensional or 1-D array has one index.

# 1-D array, shape (n,): n elements indexed [0] through [n-1]

# NumPy routines which allocate memory and fill arrays with value
a = np.zeros(4);                print(f"np.zeros(4) :   a = {a}, a shape = {a.shape}, a data type = {a.dtype}")
a = np.zeros((4,));             print(f"np.zeros(4,) :  a = {a}, a shape = {a.shape}, a data type = {a.dtype}")
a = np.random.random_sample(4); print(f"np.random.random_sample(4): a = {a}, a shape = {a.shape}, a data type = {a.dtype}")
# np.zeros(4) :   a = [0. 0. 0. 0.], a shape = (4,), a data type = float64
# np.zeros(4,) :  a = [0. 0. 0. 0.], a shape = (4,), a data type = float64
# np.random.random_sample(4): a = [0.47223374 0.62178327 0.21759447 0.79486352], a shape = (4,), a data type = float64


# NumPy routines which allocate memory and fill with user specified values
a = np.array([5,4,3,2]);  print(f"np.array([5,4,3,2]):  a = {a},     a shape = {a.shape}, a data type = {a.dtype}")
a = np.array([5.,4,3,2]); print(f"np.array([5.,4,3,2]): a = {a}, a shape = {a.shape}, a data type = {a.dtype}")


#vector indexing operations on 1-D vectors
# Create an array from 0 to 4 (exclusive)
arr1 = np.arange(5)  # [0, 1, 2, 3, 4]

# Create an array from 2 to 9 (exclusive) with a step of 2
arr2 = np.arange(2, 10, 2)  # [2, 4, 6, 8]

# Create an array from 0 to 1 with 5 evenly spaced values
arr3 = np.arange(0, 1, 0.2)  # [0. , 0.2, 0.4, 0.6, 0.8]

a = np.arange(10)
print(a)

#access an element
print(f"a[2].shape: {a[2].shape} a[2]  = {a[2]}, Accessing an element returns a scalar")

# access the last element, negative indexes count from the end
print(f"a[-1] = {a[-1]}")

#indexes must be within the range of the vector or they will produce and error
try:
    c = a[10]
except Exception as e:
    print("The error message you'll see is:")
    print(e)