Please take a look at lib/VariationalDiffusion.jl.
I wanna assume the VP-SDE $\epsilon_\theta (x,t)$ given as a diffusion model according to this paper: https://arxiv.org/abs/2011.13456.

It should wrap some Lux.jl model as $\epsilon_\theta (x,t)$

From there it should use RED-Diff or some other method to build an implicit

Please in the same manner insert the information into the obsidian vault.

My idea was:

# Implicit RED-Diff
## Error calculation
Let $P_{in}, P_{out}, P_{latent}\in\{0,1\}^{n\times n}$ be diagonal selection matrices with

$$P_{in} + P_{out} + P_{latent} = \mathrm{Id}, \qquad P_{in}P_{out} =P_{in}P_{latent} =P_{out}P_{latent} = 0$$

so that $x_"in" = P_"in"x$, $x_"out" = P_"out" x$. i.e. **we choose input and output channels**. With that assemble:

$$ P = \rho_{in}P_{in} + \rho_{out}P_{out} + \rho_{latent}P_{latent} $$

We calculate the Energy/Error of the system as:
$$E(x_0,x) = \mathbb{E}_{t, \epsilon}[\omega(t) \| \epsilon_\theta(\alpha_t x + \sigma_t \epsilon, t) - \epsilon \|_2^2] + ½\|P(x_0-x)\|^2$$

from that we can get a gradient for the Error via the [RED-Diff](https://arxiv.org/abs/2305.04391) machinery.

Alternatively use  [ProxDM](https://arxiv.org/pdf/2507.08956)  to achieve a prox.