# Implicit Diffusion Model based on RED-Diff

**Implicit** i.e. replacing learning functions by learning relations [see here](https://implicit-layers-tutorial.org/)
| Explicit Machine Learning | Implicit Learning |
|---|---|
| Approximator: functions $f_\theta:X\rightarrow Y$ | Approximator: relations $R_\theta\subset X_1\times ...\times X_n$ |

How can we learn this?
- Introduce Error/Energy space $E$ (assume multivariate)
- Learn with the function:
$$ r_\theta: X_1\times ...\times X_n \rightarrow E $$
where:
$$ (x_1,...,x_n)\in R_\theta\; :\Longleftrightarrow\; r_\theta(x_1,...,x_n) \approx 0 $$
 

## Simple Example:

| Aspect | Explicit | Implicit |
|---|---|---|
| **Approximator** | multivariate polynomials | algebraic varieties |
| **Inference** | Forward evaluation | Rootfinding |
| **Backpropagation** | Reverse mode automatic differentiation | Implicit function theorem / differential algebra |
| **Universal approximation theorem** | compact continuous functions via Weierstraß theorem | compact smooth manifolds via Nash–Tognoli theorem |
| **Well-posedness** | Always single-valued | May be multi-valued or have no solution/output only closest point to variety, instead of point on variety |
| **Loss formulation** | $\|f_\theta(x) - y\|^2$ | $\|r_\theta(x_1,..., x_n)\|^2$  |
| **Symmetry handling** | fixed unidirectional output direction | Symmetric: no distinguished input/output |
| **Computational cost of inference** | Cheap | Expensive (Newton's method, etc) |
| **Resulting Layer connections** | Directed Acyclic Graph | Arbitrary connected graph |

To a limited degree Implicit Layers, Deep Equilibrium Networks (DEQs), NeuralODEs fullfill this.

# Implicit RED-Diff
## Error calculation
Let $P_{in}, P_{out}, P_{latent}\in\{0,1\}^{n\times n}$ be diagonal selection matrices with

$$P_{in} + P_{out} + P_{latent} = \mathrm{Id}, \qquad P_{in}P_{out} =P_{in}P_{latent} =P_{out}P_{latent} = 0$$

so that $x_"in" = P_"in"x$, $x_"out" = P_"out" x$. i.e. **we choose input and output channels**. With that assemble:

$$ P = \rho_{in}P_{in} + \rho_{out}P_{out} + \rho_{latent}P_{latent} $$

We calculate the Energy/Error of the system as:
$$E(x_0,x) = \mathbb{E}_{t, \epsilon}[\omega(t) \| \epsilon_\theta(\alpha_t x + \sigma_t \epsilon, t) - \epsilon \|_2^2] + ½\|P(x_0-x)\|^2$$

from that we can get a gradient for the Error via the [RED-Diff](https://arxiv.org/abs/2305.04391) machinery.

Alternatively use  [ProxDM](https://arxiv.org/pdf/2507.08956)  to 