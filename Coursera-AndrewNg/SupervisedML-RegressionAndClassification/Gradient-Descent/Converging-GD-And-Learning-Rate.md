# How to check if Gradient Descent algorithm is converging for our model?

## Learning Curve

- Cost vs Iteration graph
- Plotting a graph after simultaneous updates of the w and b in the algorithm.

- After some iterations let's say 500, the graph gets flattened out then gradient descent is converged.

    ![alt text](images/converged-gd-image.png)


- If the learning curve is zig-zag then something is wrong with the gradient descent algorithm.
- Either learning rate is too large or there is some bug in the code.

# Learning Rate (α)

- If we choose very small α, then gradient descent takes a lot of iterations to converge.
- If we choose very big α, then gradient descent curve will show zig-zag patterns, and will overshoot the expected outcome.

### Tips for choosing α

- While choosing α, we should choose α small enough that the cost function should decrease with each iteration.

- After choosing α, we can plot a learning curve for first few iterations and check if the cost function is decreasing.

- We can choose one α very small, and one very α, say 3 times the small one.
- And find out if small one is decreasing slow, and if large one if making the learning curve zig-zag.
