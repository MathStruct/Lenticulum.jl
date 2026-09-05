# Lenticulum.jl (Under Development)

[![Build Status](https://github.com/DanielBoigk/Lenticulum.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/DanielBoigk/Lenticulum.jl/actions/workflows/CI.yml?query=branch%3Amain)

**Implicit** i.e. replacing learning functions by learning relations [see here](https://implicit-layers-tutorial.org/)
| Explicit Machine Learning | Implicit Learning |
|---|---|
| Approximator: functions $f_\theta:X\rightarrow Y$ | Approximator: relations $R_\theta\subset X_1\times ...\times X_n$ |

How can we learn this?
- Introduce Error/Energy space $E$ (assume multivariate)
- Learn with the function:

$$
r_\theta : X_1 \times \cdots \times X_n \rightarrow E
$$

where:

$$
(x_1,\ldots,x_n)\in R_\theta
\;:\Longleftrightarrow\;
r_\theta(x_1,\ldots,x_n) \approx 0
$$

 
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

## What is the difference to Lux.jl?

Mainly three differences:
- Lux.jl is build around a categorical concept which is called a [parametric Lens](https://arxiv.org/html/2103.01931v2#S2) (Definition 2.5), this structures aside from initialization requires definition of two functions get (i.e. inference) and set (i.e. backpropagation). we however require our factors (how we call our layers) to be a [parametrized Statistical game](https://arxiv.org/html/2503.18608v2#S5) (Definition 27), which first requires our program to extract or assemble parametric Lenses before they can be used.
- Layer connections in Lux.jl (i.e. the way the layers are wired) need to be a directed acyclic graph (DAG). We end up with pretty much any weakly connected directed graph. Nonetheless factors internally use Lux.jl.
- Message passing in Lux.jl is trivial. we can use the default dense message passing scheme which always passes between all factors, however for performance reason we might consider sparsifying our message passing scheme.

## Factor graphs and Message passing
A factor graph is a bipartite graph consisting of variables and factors. This concept is well known in SLAM engineering and is a graphical model for bayesian modeling.
- Our training data are fixed variables in our factor graph graph.
- Optimizers (like ADAM, ...) and Losses become nonparametric factors in our computation graph.
- A factor has multiple channels by which it can be connected to variables. Passing a message through a factor requires choosing which channels are input and which are output. Only then functions are assembled and the message is passed.

We therefore have to additionally resolve the graph handling and can not rely on traditional Data handling like done with MLUtils.jl.
