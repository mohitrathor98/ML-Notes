### Gradient Descent

- A machine learning algorithm to determine minimum values of a function.

- Idea is to change the value of w and b so that we reach near the minimum of the cost function J(w, b).

- Works by adjusting the parameters in the direction of negative gradient(i.e, descent) of the function.

- By taking the steps in the negative descent direction, we eventually reach the minimum value of the function.


### Gradient Descent Algorithm

- We take the parameter w and reduce it by a small amount which is alpha times derivative of cost function.

- Similarly, for parameter b we do the same thing.

        - Alpha is the learning rate.
        - We reduce the alpha times derivative of cost function until the convergence.

***Notes:***

- <u>***Point of Convergence***</u>: When values of w and b doesn't changes much on repeating the descent algorithm.

- <u>***Simultaneous Update***</u>: Both w and b should be updated simultaneously. This means that <em>we should not update b with the updated value of w. We can make use of temp variables and then updated w and b at the end of steps.</em>


![Reducing parameters](images/image-Gradient1.png)
     


## Gradient Descent Intuition

### Derivative Intuition

- Let's check the behavior by taking b = 0 for better understanding.

- The derivative d(J(w))/dw represents the slope of the graph J(w) vs w at the position of current w.

- Now the gradient descent algorithm will subtract the derivative from w.

- If the derivative or slope is positive, the w will decrease.

- If the derivative or slope is negative, the w will increase.

- As shown in the below graph, in either cases, J(w) decreases.

***Note***

- If w or J(w) is already at a local minima, then w won't change because derivative part will be zero.


![Derivative intuition](images/image-derivative.png)


### Learning Rate - Alpha

- if learning rate is too small, gradient descent will be very slow.

- if learning rate is too large, gradient descent may:

    - Overshoot, i.e, never reaches minimum.
    - Or Fail to converge or get diverged.