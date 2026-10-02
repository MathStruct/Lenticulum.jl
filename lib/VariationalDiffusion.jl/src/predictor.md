#implementation

> `NoisePredictor` wraps a Lux model as $\varepsilon_\theta(x,t)$ and derives from it the two
> quantities the rest of the package needs: the **score** and the **denoiser**.

> Sources: code: `predictor.jl`
>
> Theory (CT-ML wiki): [Parametric Lens](https://mathstruct.org/CategoryTheory-ML-Wiki/Parametric-Lens) · [Lens](https://mathstruct.org/CategoryTheory-ML-Wiki/Lens)

## 1. LuxCore, not Lux

The prompt asks the package to "wrap some Lux.jl model". It does — and `Lux` is not a
dependency. Any Lux layer is a `LuxCore.AbstractLuxLayer`, and the wrapper only ever calls
`LuxCore.apply(model, input, ps, st)`. So `Lux`, `Zygote`, `Optimisers` and the rest stay out
of the dependency tree, exactly as the parent package depends on `LuxCore` rather than `Lux`
([[Lux as a Parametric Lens]] §"What is kept unchanged").

`NoisePredictor <: LuxCore.AbstractLuxWrapperLayer{:model}`, which is the LuxCore trick for
"my parameters *are* my child's parameters, with no extra nesting". So

```julia
LuxCore.setup(rng, NoisePredictor(unet, VPSDE())) == LuxCore.setup(rng, unet)
```

and wrapping costs nothing. This is the one place where the project's decision to mirror the
LuxCore API name-for-name pays off directly rather than aspirationally.

## 2. Three quantities, one network

$$
\underbrace{\varepsilon_\theta(x,t)}_{\texttt{epsilon}}
\qquad
\underbrace{s_\theta = -\frac{\varepsilon_\theta}{\sigma_t}}_{\texttt{score}}
\qquad
\underbrace{\hat x_0 = \frac{x - \sigma_t\varepsilon_\theta}{\alpha_t}}_{\texttt{denoise}}
$$

The second and third are **derived, not learned**, and that they are derived is the whole
reason a noise predictor can serve as a prior rather than only as a sampler.

The score identity is exact for the perturbation kernel: with
$x_t = \alpha_t x_0 + \sigma_t\varepsilon$,

$$\nabla_{x_t}\log p_{0t}(x_t\mid x_0) = -\frac{x_t-\alpha_t x_0}{\sigma_t^2} = -\frac{\varepsilon}{\sigma_t}$$

so a perfect $\varepsilon_\theta$ is a perfect score. The denoiser is Tweedie's formula, and

> [!important] Tweedie is exact, and the test suite proves it
> For Gaussian data $x_0\sim\mathcal{N}(0,v_0)$ the closed-form posterior mean is
> $\mathbb{E}[x_0\mid x_t] = \alpha_t v_0 x_t/(\alpha_t^2v_0+\sigma_t^2)$, and `denoise`
> reproduces it **to floating point**, for every $t$ tested. Tweedie's formula turns a noise
> predictor into an MMSE denoiser; that is the precise sense in which a diffusion model is a
> prior, and it is what the RED-Diff regulariser scores against.

## 3. The `input` adapter

Diffusion architectures disagree about how $t$ arrives: a second positional argument, a
sinusoidal embedding, a concatenated channel, a `NamedTuple`. The wrapper takes an `input`
function, defaulting to `(x, t) -> (x, t)`, and **knows nothing about time embeddings** —
that is the model's business.

The cost is that a wrong `input` fails inside the user's model with the user's error message
rather than here with a good one. There is no way to validate it without knowing the
architecture, so it is not validated.

## 4. Implementation difficulties

### 4.1 Forward-only where it can be, any AD backend where it cannot

`epsilon` is the only place the network is evaluated forwards, and RED-Diff never needs a
reverse pass through `model`. That is RED-Diff's stop-gradient, promoted to an architectural
property: sampling and RED-Diff work with no AD at all. `reddiff.md` §5.

The implicit learner is different. Its Newton steps want the input Jacobian, and its backward
pass wants a parameter VJP. Both go through the `ad` field — any ADTypes object — and are
supplied by package extensions, so the package still has no AD dependency of its own. `epsilon`
itself calls the hook `_apply(ad, …)`, which a backend may replace by a compiled forward pass
(`ad = AutoReactant()`). See [[backends]]. Training $\varepsilon_\theta$ by denoising score
matching is ordinary Lux training (`examples/circle_mlp.jl`).

### 4.2 The 1/σ_t in `score` is a real singularity

`score` divides by $\sigma_t$, so it is unusable at $t=0$ and inaccurate for small $t$ even
with the `tmin` floor. Nothing in this package calls `score` — the RED-Diff gradient is
expressed in $\varepsilon$-space precisely to avoid it — so it exists for completeness and
for anyone building a sampler on top. If you call it near $t=0$, that is on you.

### 4.3 `denoising_loss` returns a scalar, so it cannot be a vector energy

[[Scalar and Multivariate Energy]] argues that a factor should expose its *vector* residual,
because the implicit function theorem needs the Jacobian. `denoising_loss` collapses to
`sum(abs2, ε̂ - ε)` immediately.

The vector residual is right there — $\varepsilon_\theta - \varepsilon$ — and returning it
would let `SquaredNorm` do the scalarisation and would make the diffusion factor's energy
space $\mathbb{R}^n$ rather than $\mathbb{R}$. It is not done because RED-Diff's gradient is
already in closed form and the Jacobian is exactly what the method refuses to compute; the
vector form would be honest bookkeeping with nothing downstream to consume it. **Recorded as
a genuine inconsistency with the vault's stated design, with a reason.**

Related: [[schedule]], [[reddiff]], [[factor]], [[backends]], [[The VP-SDE]],
[[Lux as a Parametric Lens]], [[Scalar and Multivariate Energy]]
