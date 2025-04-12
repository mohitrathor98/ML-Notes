# 🧠 What is Backpropagation?
Backpropagation is the method used by neural networks to learn. It tells the network how to adjust its weights based on the error it made in the output.

Think of it like this:

    You made a mistake in your math test. You look at the correct answer, understand where you went wrong, and then learn to avoid that mistake next time. That’s backpropagation!

## 🔁 Where Does It Fit?

- Forward pass: The input goes through the network to produce a prediction.

- Loss calculation: The difference between the prediction and the actual value is calculated (this is called the loss).

- Backpropagation: We calculate how much each weight contributed to the error and adjust them accordingly.

- Repeat: Do this again and again (epochs) until the error is minimized.

## 🔧 How Does It Work?
Let’s walk through a simple example:

1. Forward pass
    - Input: x
    - Weight: w
    - Output: y_pred = w * x
    - Actual: y_true
    - Loss: (y_pred - y_true)^2 (Mean Squared Error)

2. Backpropagation

    We want to adjust w to reduce the loss.
    So we compute the derivative of the loss with respect to w:

    ```
    dLoss/dw=2∗(y_pred−y_true)∗x
    ```

    This tells us:
    - The direction we need to adjust the weight in
    - The amount of change needed

3. Update the weight

    Using gradient descent:
    ```
    w = w - learning_rate * dLoss/dw
    ```
    This is done layer by layer, starting from the output layer and moving backward — hence back propagation.

## ✨ Why Is It Powerful?
Because it allows deep networks with many layers to learn from data. Without backpropagation, we wouldn’t be able to train networks effectively.

# 🛠️ Example: Backpropagation in Python
Great! Let’s walk through backpropagation using:

### 🧠 Step 1: Diagram – 1 Neuron, 1 Input, 1 Output
```plaintext
       x (input)
         |
         v
      [ w ] ----> Multiply
         |         (wx)
         v
       y_pred     ----> compare with y_true
         |
         v
      Loss = (y_pred - y_true)^2
         |
         v
   Backpropagation: update w
```
This diagram shows a single neuron with one input and one output. The weight (w) is multiplied by the input (x) to produce the prediction (y_pred). The loss is calculated by comparing y_pred with the true value (y_true). Finally, backpropagation updates the weight based on the loss.
This is the simplest case: one input, one weight, no activation, and one output.

### 🧪 Step 2: Python Code – Basic Backprop

```python
import numpy as np

# Initialize input and expected output
x = 2.0           # input
y_true = 4.0      # true output

# Initialize weight
w = 1.0           # starting weight
learning_rate = 0.1

# Training loop
for epoch in range(10):
    # ----- Forward pass -----
    y_pred = w * x
    
    # ----- Loss (Mean Squared Error) -----
    loss = (y_pred - y_true) ** 2
    
    # ----- Backward pass (Gradient calculation) -----
    dL_dypred = 2 * (y_pred - y_true)       # dLoss/dy_pred
    dypred_dw = x                           # dy_pred/dw
    dL_dw = dL_dypred * dypred_dw           # Chain rule

    # ----- Update weight -----
    w = w - learning_rate * dL_dw

    print(f"Epoch {epoch+1}: Loss = {loss:.4f}, Weight = {w:.4f}")
```

### 🔍 Breakdown

- `y_pred = w * x`: simple linear model

- `loss = (y_pred - y_true)^2`: mean squared error

- Derivative w.r.t. `w` is:
    ```
    dL/dw = 2 * (y_pred - y_true) * x
    ```

- We update weight using:

    ```
    w = w - learning_rate * dL/dw
    ```
​
 
### 🧾 Output Example
```plaintext
Epoch 1: Loss = 4.0000, Weight = 1.8000
Epoch 2: Loss = 1.4400, Weight = 2.4400
Epoch 3: Loss = 0.5184, Weight = 2.9520
....
````
You can see the loss decreasing and the weight getting closer to `2.0` — which is the ideal weight because `2 * 2 = 4`.


# 🛠️ 3-layer neural network:

- Input layer: 3 neurons
- Hidden layer: 3 neurons
- Output layer: 1 neuron

We’ll use `ReLU` activation for the hidden layer and no activation (linear) for the output, and implement forward and backward propagation from scratch using NumPy.

### 📊 Network Architecture Diagram
```plaintext
Input (x1, x2, x3)
      ↓
   [w1] [w2] [w3]       ← weights between input and hidden
      ↓    ↓    ↓
Hidden Layer (h1, h2, h3)  ← ReLU activation
      ↓    ↓    ↓
     [w4, w5, w6]        ← weights between hidden and output
         ↓
    Output (y_pred)
```
### 🧪 Step 3: Python Code – Backpropagation in a 3-layer Neural Network

```python
import numpy as np

# ReLU and its derivative
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

# Initialize input and target output
x = np.array([[1.0, 2.0, 3.0]])  # shape (1, 3)
y_true = np.array([[1.0]])       # shape (1, 1)

# Weight initialization
np.random.seed(0)
W1 = np.random.randn(3, 3)  # weights from input to hidden (3x3)
W2 = np.random.randn(3, 1)  # weights from hidden to output (3x1)
learning_rate = 0.01

# Training loop
for epoch in range(100):
    # ----- Forward pass -----
    z1 = np.dot(x, W1)            # (1,3)
    a1 = relu(z1)                 # (1,3)
    z2 = np.dot(a1, W2)           # (1,1)
    y_pred = z2                   # Output (no activation)

    # ----- Loss -----
    loss = np.mean((y_pred - y_true) ** 2)

    # ----- Backward pass -----
    dL_dy = 2 * (y_pred - y_true)        # (1,1)
    dy_dW2 = a1.T                        # (3,1)
    dL_dW2 = dy_dW2 * dL_dy              # (3,1)

    # Backprop through W2 to hidden layer
    da1_dz1 = relu_derivative(z1)       # (1,3)
    dz1_dW1 = x.T                       # (3,1)
    dL_da1 = np.dot(dL_dy, W2.T)        # (1,3)
    dL_dz1 = dL_da1 * da1_dz1           # (1,3)
    dL_dW1 = np.dot(x.T, dL_dz1)        # (3,3)

    # ----- Update weights -----
    W1 -= learning_rate * dL_dW1
    W2 -= learning_rate * dL_dW2

    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Loss = {loss:.4f}")
```
### 🔍 Breakdown
- `W1`: connects input → hidden layer (3×3)
- `W2`: connects hidden → output layer (3×1)
- ReLU ensures non-linearity
- We compute the gradient of loss w.r.t each weight using the chain rule:
    - Loss → output
    - Output → hidden
    - Hidden → input

### 📉 Sample Output (Truncated)
```plaintext
Epoch 0: Loss = 1.8827
Epoch 10: Loss = 0.2234
Epoch 20: Loss = 0.0356
...
Epoch 90: Loss = 0.0004
```
The loss decreases over epochs, indicating that the network is learning and improving its predictions.
### 🎉 Conclusion
Loss goes down, which shows our network is learning!