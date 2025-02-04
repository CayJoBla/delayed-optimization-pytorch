# Improving Optimization using Time Delays
This project deals with adding time delays to a variety of gradient-based optimization methods, given some objective function and its gradient.
Through experimentation, we have discovered that introducing certain time delays into optimization algorithms, especially the Adam optimizer, have the potential to significantly improve the performance of the optimizer. 
Perhaps more surprising is that this improvement often scales with dimension, resulting in better performance relative to the undelayed optimizer under high-dimensional objective functions.
Furthermore, this algorithm does not affect the leading-order computational complexity of the optimization method, and the spatial complexity scales linearly with the length of the largest time delay.

## Interfacing with Pytorch
Specifically, this repository contains a wrapper for Pytorch optimization algorithm implementations that will apply a variety of different time delays to the optimizer.

## Installing DeepOBS
For this project, we use the development version of the DeepOBS benchmarking library for its pytorch support on a variety of benchmarking tasks.

For the editable version of DeepOBS:
```bash
pip install -e 'git+https://github.com/fsschneider/DeepOBS.git@develop#egg=deepobs'
```
For this, it is important that you use the 1.4.3 version of the `bayesian-optimization` package, as the latest version is not compatible.
