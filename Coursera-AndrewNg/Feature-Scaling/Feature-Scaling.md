# Feature Scaling

## When to Scale?

- If any feature has very high or very low range of values then we want to scale.
- We need data to be around [-1, 1] range. Else, we can scale.

        Examples:
        [0, 3] is ok
        [-2, 0.5] is ok
        [-100, 100] needs to be scaled.
        [-0.0001, 0.0001] needs to be scaled.

## Introduction

- Suppose we are given a dataset of houses, where x1 is size in sq feet and x2 is number of bedrooms.
- lets say x1 = 2000, x2 = 5. Expected output y = 500k dollars.

        y-hat = w1*x1 + w2*x2 +b

        let's have b = 50.

- <strong>If we choose w1 high and w2 low, then w1.x1 will increase the prediction value a lot. And w2.x2 won't have much effect in the prediction.
- If we choose w1 low and w2 high, then both w1x1 and w2x2 can have similar effect on the prediction.</strong>

        if w1 = 50, w2 = 0.1 ==> y-hat = 50 * 2000 + 0.1 * 5 + 50 = 100,050,500 dollars

        and if w1 = 0.1, w2 = 50 ==> y-hat = 500,000 dollars. ==> Exactly what we expect.


- In these scenarios, we need to choose the initial values of the parameters very cautiously.
- Otherwise, Gradient Descent algorithm will take a very long time to converge to the expected values (minima).

- ***Hence, in these scenarios, it's better to scale X1 and X2 values to have similar ranges.***

## How to do feature scaling on given data set?

- We can divide the whole X1 or X2 to get to the similar ranges.

        Let's say X1 has range [300, 2000] and X2 has [0, 5].
        if we want to scale we can do,

        x1-scaled = x1/2000  and  x2-scaled = x2/5
        then 
        x1-scaled will have range [0.15, 1] and x2-scaled will have [0, 1]



## Mean Normalization Feature Scaling

- Another method to scale the data.
- We scale our features, so that they are centered around zero. (Can have -ve or +ve values)

#### Calculation of Mean Normalization
    
        Ex: 300 <= X1 <= 2000  and  0 <= X2 <= 5

- First, find average of the dataset for feature i (µᵢ)

- Then, we normalize the feature using below formula

        [Xi = (Xi - µᵢ) / (max(Xi) - min(Xi))]


## Z-Score Normalization

- Need to calculate standard deviation of features.

- Standard Deviation: Measures how far data points are from the mean in the dataset.

#### Calculation of Z-score Normalization

- Calculate mean of the feature:(µᵢ)

- Calculate Standard Deviation: (σᵢ)

- Scale using this formula

        [Xᵢ = (Xᵢ-µᵢ) / σᵢ]