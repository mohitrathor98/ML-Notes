# Multi-Label Classification.

- Used in scenrios where output label Y can have multiple value.
- Ex: Y = [1, 0, 1] ==> Three values coming from the model.

## Scenario

- Detecting if in given picture, we have car, bus and pedestrian?

- We can solve this by:

    1)  Having three different models which predicts the three outputs.
    2)  Train one neural network with three outputs.

        - At the end of the neural net, add one layer with as many neurons as the number of outputs required.
        - Those neurons can have sigmoid activation. 
        - Now, final output will be a vector with all three labels.