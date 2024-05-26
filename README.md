# DPSGD with Opacus
In this repository, I conducted experiments using [Opacus](https://opacus.ai/), a library that enables training PyTorch models with differential privacy. 

## About DP-SGD
The algorithm was introduced in the paper [Deep Learning with Differential Privacy](https://arxiv.org/abs/1607.00133). There are three major differences compared to standard SGD:
- `Norm Clipping`: Each gradient is clipped based on its L2 norm. By clipping the gradients, no individual data point can significantly affect the model's parameters, thereby helping to preserve privacy.
- `Add Noise`: Controlled noise is added after averaging each lot of gradients. This ensures differential privacy by making it harder for an external observer to ascertain whether a specific example was part of the training data.
- `Moment Accountant`: It tracks higher-order moments of the privacy loss distribution and provides more accurate tracking of cumulative privacy loss over multiple iterations. This helps ensure that the overall privacy budget is not exceeded, maintaining the desired level of privacy.

## Experiments and results
The objective is to study the impact of the privacy guarantee ($\epsilon$) and the norm-clipping parameter ($C$) on the model’s accuracy and utility. Results on the MNIST dataset indicate that:
- Increasing the privacy guarantee ($\epsilon$) can reduce the model's performance.
- Changing the norm-clipping parameter ($C$) does not significantly affect accuracy but results in a smoother training curve.

|C|EPS|Acc after 20 epochs|
|--|--|--|
|1.0|5|0.5677|
|1.0|10|0.5938|
|1.0|50|0.6369|
|5.0|10|0.5923|
|10.0|10|0.5900|

Graphs are shown in the file `debug.ipynb`

## Todo
- Apply DP-SGD to Federated Learning (FL)
- Compare different noise distributions, such as Gaussian and Laplace
- Explore different mechanisms for privacy accounting, such as Rényi Differential Privacy (RDP) and Gaussian Differential Privacy (GDP).
