# Implicit Variational Diffusion Models

we sample some stochastic forward process as:

$$
x_t = \alpha_t x_0 + \sigma_t \epsilon
$$

From here on we can calculate a variational error as:

$$
R(x_0) = \mathbb{E}_{t, \epsilon}[\omega(t) \| \epsilon_\theta(\alpha_t x_0 + \sigma_t \epsilon, t) - \epsilon \|_2^2]
$$
