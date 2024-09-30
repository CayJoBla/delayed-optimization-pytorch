# Improving Optimization using Time Delays
This project deals with adding time delays to a variety of gradient-based optimization methods, given some objective function and its gradient.
Through experimentation, we have discovered that introducing certain time delays into optimization algorithms, especially the Adam optimizer, have the potential to significantly improve the performance of the optimizer. 
Perhaps more surprising is that this improvement often scales with dimension, resulting in better performance relative to the undelayed optimizer under high-dimensional objective functions.
Furthermore, this algorithm does not affect the leading-order computational complexity of the optimization method, and the spatial complexity scales linearly with the length of the largest time delay.

## Interfacing with Pytorch
Specifically, this repository contains a wrapper for Pytorch optimization algorithm implementations that will apply a variety of different time delays to the optimizer.
