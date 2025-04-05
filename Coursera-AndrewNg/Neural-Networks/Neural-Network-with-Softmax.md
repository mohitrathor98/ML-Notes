# Neural Network with Softmax function

- If we want to predict more than two outputs then we will have an extra layer at the end
of our current nueral network.
- Which is called `Softmax Output Layer`.

- In Softmax layer, activation of each of the neuron is dependent upon the activation function values of all other neurons in the layer.

## Tensorflow implmentation (Flawed)

- We use `softmax` as activation for the final layer, which contains as many neurons as the number of expected outputs.

- Loss function will be `SparseCategoricalCrossentropy`.

- Sparse Categorical refers that 'Y' is still classified into categories and it still can take only one of the expected values.

***NOTE:*** Below code works, however, this is not optimized.

```
    import tensorflow as tf
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import Dense

    model = Sequential([
        Dense(units=25, activation='relu'),
        Dense(units=15, activation='relu),
        Dense(units=10, activation='softmax')
    ])

    from tensorflow.keras.losses import SparseCategoricalCrossentropy
    model.compile(loss= SparseCategoricalCrossentropy())
```

## Improved implementation

- Division of small floating point numbers like 1/100000, can cause round-off errors.
- It can impact the overall performance of the model.

### Solution

- Instead of using `Softmax` in output layer, we use `linear` activation function. 
- And, use below code:
```
    model.compile(loss=SparseCategoricalCrossentropy(from_logits=True))
```

- logits is Z (which is W.X + b) 

### Fit and predict the label

- After compile:
```
    model.fit(X, Y, epochs=100)
    logit = model(X)
    f_x = tf.nn.sigmoid(logit)
```