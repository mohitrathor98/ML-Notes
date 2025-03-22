## How to implement a basic NN model using Tensorflow?

Let's take a coffee bean example.

- Feature Vector (x-vector) consists of temperature and duration of roasting.
- The model will predict if coffee is good or bad.

![Coffee-Roating](images/Coffee-Roasting.png)

### First Layer

![First-Layer](images/first-NN-layer-coffee-roasting.png)

- Dense: Type of neural net library provided by tensorflow. It returns a function.
- Applying the returned function on vector X, we get a1 (activation of first layer)

### Second Layer

![Second-Layer](images/scond-NN-layer-coffee-roasting.png)

- Similarly we get value of a2 which is the final activation of our model.
- Now, using the final activation value we can predict if coffee is good or bad.

```
    if a2 >= 0.5:
        yhat = 1
    else:
        yhat = 0
```