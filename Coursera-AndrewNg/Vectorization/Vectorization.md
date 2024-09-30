## Vectorization

- A technique in computing where operations are performed on multiple pieces of data a once.

- Instead of operating on each individual elements.
- Do not iterate on each element, and perform operations on the entire vector at once.

- Greatly speeds up the computations on large datasets.

- It processes data in parallel.
- Hence, make use of CPUs and GPUs multiple processing units.

### Creating vectors in python

        if we have
        w-vector = [w1 w2 w3]
        x-vector = [x1 x2 x3]

        using numpy as np
        w = np.array([w1, w2, w3])
        x = np.array([x1, x2, x3])

        print(w[0]) # prints the value of w1

### Without vectorization code to compute y-hat of multiple features


        f = w[0]*x[0] + w[1]*x[1] + w[2]*x[2] + b

        or
        
        f = 0
        for j in range(0, n):
            f += w[j]*x[j]
        f += b

### With Vectorization

        f = np.dot(w, x) + b
