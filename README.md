# Improving Optimization using Time Delays
This project deals with adding time delays to a variety of gradient-based optimization methods, given some objective function and its gradient.
Through experimentation, we have discovered that introducing certain time delays into optimization algorithms, especially the Adam optimizer, have the potential to significantly improve the performance of the optimizer. 
Perhaps more surprising is that this improvement often scales with dimension, resulting in better performance relative to the undelayed optimizer under high-dimensional objective functions.
Furthermore, this algorithm does not affect the leading-order computational complexity of the optimization method, and the spatial complexity scales linearly with the length of the largest time delay.

## Using the time-delayed optimizer
To use the time-delayed optimizer with pre-defined functions, optimizers, and delay types, you can utilize the `OptimizationHelper` class, which allows the user to automatically run optimization, and access optimization data. An example of how to use this class is shown in the `demo.ipynb` notebook.

To instead define your own objective function or optimizer, you can use the `DelayedOptimizer` class, passing in an objective function, optimizer, and delay distribution to use in optimization. This allows the user to manually handle their use case, including control over the optimization loop itself. An example of how to use this class is also shown in the `demo.ipynb` notebook.
