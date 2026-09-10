# The Linear Gaussian Chain

> The example every factor-graph library opens with — GTSAM's `OdometryExample` — written
> against this library. A robot drives along a line, odometry links consecutive poses, one
> GPS reading lands on the last pose, and the posterior over *all* poses comes back exact.
>
> It is the smallest problem in which this library does something a `Lux.jl` pipeline
> cannot: the same factor is run in **both directions** and information travels **backwards**
> along the chain.

## 1. The model

$$
\begin{aligned}
x_1 &\sim \mathcal{N}(\mu_0, \Sigma_0) \\
x_{k+1} &= x_k + u_k + \varepsilon_k, & \varepsilon_k &\sim \mathcal{N}(0, Q_k) \\
z &= x_n + \delta, & \delta &\sim \mathcal{N}(0, R)
\end{aligned}
$$

with $z$ clamped to the observed reading $z_0$. Everything is linear and everything is
Gaussian, so the posterior is Gaussian and known in closed form — which is the point: this
is an **oracle**, not a demo.

```tikz
\usepackage{tikz}
\begin{document}
\begin{tikzpicture}[
  var/.style={circle, draw, minimum size=7mm, inner sep=0pt},
  fac/.style={rectangle, draw, fill=black!12, minimum size=5mm, inner sep=1.5pt},
  dat/.style={rectangle, draw, fill=black!55, text=white, minimum size=5mm, inner sep=1.5pt},
  every node/.style={font=\small}
]
\node[fac] (pi)  at (0,0)   {$\pi$};
\node[var] (x1)  at (1.4,0) {$x_1$};
\node[fac] (o1)  at (2.8,0) {$\mathrm{odo}_1$};
\node[var] (x2)  at (4.2,0) {$x_2$};
\node[fac] (o2)  at (5.6,0) {$\mathrm{odo}_2$};
\node[var] (x3)  at (7.0,0) {$x_3$};
\node[fac] (gp)  at (7.0,-1.4) {$\mathrm{gps}$};
\node[var] (z)   at (7.0,-2.8) {$z$};
\node[dat] (d)   at (8.7,-2.8) {$z_0$};
\draw (pi)--(x1) (x1)--(o1) (o1)--(x2) (x2)--(o2) (o2)--(x3)
      (x3)--(gp) (gp)--(z) (z)--(d);
\end{tikzpicture}
\end{document}
```

Circles are variables (wires), squares are factors. There is no cycle, so this is a
`istree(g) == true` graph and belief propagation is **exact** — see [[Factor Graphs]] for the
two distinct acyclicity notions and why it is `istree` and not `isdag` that matters here.

## 2. Writing it

The whole model is built from **one** factor type used three times.

```julia
using Lenticulum, LenticulumCore, Mycelium

μ₀, Σ₀ = [0.0], fill(1.0, 1, 1)             # prior on x₁
u      = ([2.0], [2.0])                     # odometry measurements
Q      = (fill(0.25, 1, 1), fill(0.25, 1, 1))
R      = fill(0.5, 1, 1)                    # GPS noise
z₀     = [4.5]                              # the GPS reading
Id     = fill(1.0, 1, 1)

b = GraphBuilder()
for v in (:x1, :x2, :x3, :z); variable!(b, v, 1); end

factor!(b, :prior, GaussianPrior(:x1, μ₀, Σ₀))
factor!(b, :odo1, GaussianFactor(1 => 1; noise = Q[1], channels = (:x, :y)))
factor!(b, :odo2, GaussianFactor(1 => 1; noise = Q[2], channels = (:x, :y)))
factor!(b, :gps,  GaussianFactor(1 => 1; noise = R,    channels = (:x, :y)))
factor!(b, :data, DataFactor(:z, z₀))

connect!(b, :prior, :x1, :x1; direction = Bidirectional())
connect!(b, :odo1, :x, :x1; direction = Bidirectional())
connect!(b, :odo1, :y, :x2; direction = Bidirectional())
connect!(b, :odo2, :x, :x2; direction = Bidirectional())
connect!(b, :odo2, :y, :x3; direction = Bidirectional())
connect!(b, :gps,  :x, :x3; direction = Bidirectional())
connect!(b, :gps,  :y, :z;  direction = Bidirectional())
connect!(b, :data, :z, :z;  direction = Emitting())

g = validate(build(b))

ps = (prior = NamedTuple(), odo1 = (A = Id, b = u[1]), odo2 = (A = Id, b = u[2]),
      gps = (A = Id, b = [0.0]), data = NamedTuple())
st = (prior = NamedTuple(), odo1 = NamedTuple(), odo2 = NamedTuple(),
      gps = NamedTuple(), data = NamedTuple())

marg, report, st, store = infer!(g, tree_schedule(g), ps, st)
```

and that is the whole program. `marg.x1`, `marg.x2`, `marg.x3` are
`GaussianBelief`s; `marg.z` is the `DiracBelief` the clamp put there.

### The odometry factor is the Gaussian factor with $A = I$

$p(x_{k+1} \mid x_k) = \mathcal{N}(x_{k+1};\ x_k + u_k,\ Q_k)$ is
`GaussianFactor(1 => 1; noise = Q[k])` with $A = I$ and $b = u_k$. GTSAM calls this a
*between factor* and gives it its own class; here it is a **parameter choice**, and $u_k$ sits
in `ps` where a neural network's weights would. Nothing marks it as a measurement rather
than a learnable offset — see §6.

### The unary measurement factor is that factor plus a clamp

GTSAM's GPS factor is unary: it attaches to $x_3$ alone and carries $z_0$ inside itself. This
library refuses that, on the grounds of [[Everything is a Factor]]: *a variable is a wire and
has no content of its own, whereas data is evidence, and evidence is a factor.* So the
measurement becomes two nodes and an auxiliary variable,

$$
x_3 \;\longrightarrow\; \boxed{\mathrm{gps}} \;\longrightarrow\; z \;\longleftarrow\; \boxed{z_0}
$$

with `DataFactor(:z, z₀)` supplying the $\rho_{in} = \infty$ hard clamp of
[[Channels and Polarity]]. This costs one variable and buys three things:

1. the noise model $R$ and the datum $z_0$ are **separately** addressable — swap the clamp
   for a different reading and nothing else changes;
2. unclamping $z$ turns the measurement into a *prediction* with no code change (see §5);
3. the clamp edge is `Emitting()`, so [[Schedules|schedule pruning]] never even tries to
   compute a message back into it.

## 3. What comes out, and against what

| | mean | variance |
|---|---|---|
| $x_1$ | $0.25$ | $0.5$ |
| $x_2$ | $2.3125$ | $0.46875$ |
| $x_3$ | $4.375$ | $0.375$ |

The oracle is what GTSAM actually computes. Stack the poses into $\xi = (x_1,x_2,x_3)$; each
factor contributes a quadratic $\tfrac12\|G\xi - c\|^2_{Q^{-1}}$, so the joint is one
information form

$$
\Lambda \;=\; \sum_f G_f^\top Q_f^{-1} G_f,
\qquad
\eta \;=\; \sum_f G_f^\top Q_f^{-1} c_f,
\qquad
\hat\xi = \Lambda^{-1}\eta,
\quad
\Sigma = \Lambda^{-1}
$$

with $G_\pi = e_1^\top$, $G_{\mathrm{odo}_k} = e_{k+1}^\top - e_k^\top$ and
$G_{\mathrm{gps}} = e_3^\top$. That $\Lambda$ is the sparse Hessian a nonlinear factor-graph
library builds by linearisation, and its tridiagonality *is* the chain structure written
down — a loop closure is exactly what would put entries in the corners. Here it is dense and unapologetic, because being *obviously* right
matters more than being fast in an oracle.

$$
\Lambda = \begin{pmatrix} 5 & -4 & 0 \\ -4 & 8 & -4 \\ 0 & -4 & 6 \end{pmatrix},
\qquad
\eta = \begin{pmatrix} -8 \\ 0 \\ 17 \end{pmatrix}
$$

> [!important] The test asserts equality, not closeness
> `belief_cov(marg[v])[1] ≈ Σ[i,i]` holds to floating-point rounding, for every pose and at
> every chain length $n = 2 \ldots 8$ with inhomogeneous $Q_k$. On a tree, BP is not an
> approximation to elimination; it *is* elimination, reorganised.

### These are smoothed marginals

$\operatorname{Var}(x_1) = 0.5$, half its prior $\Sigma_0 = 1$. Nothing in the model tells
$x_1$ about the GPS reading directly — the information travelled

$$
z \;\to\; \mathrm{gps} \;\to\; x_3 \;\to\; \mathrm{odo}_2 \;\to\; x_2 \;\to\; \mathrm{odo}_1 \;\to\; x_1
$$

which is the **backward sweep** of `tree_schedule`, and requires each `odo` factor to be run
with $y$ observed and $x$ unobserved — the opposite of the direction it was written in. A
filter would report $\operatorname{Var}(x_1) = 1$; a smoother reports $0.5$. Getting the
smoother for free from the schedule, rather than as a second algorithm, is the payoff of
[[Messages are Inversions]].

> [!note] This is why factors are relations, not functions
> A `Lux.jl` layer has one direction and its reverse pass carries a gradient. `odo1` here has
> two directions and both carry *posteriors*. Running it backwards is not
> differentiating it — it is `assemble`-ing a different [[Bayesian Lens]] out of the same
> factor, and it is exact.

## 4. The evidence falls out too

`scalar_free_energy` on the converged store returns

$$
F \;=\; -\log p(z_0) \;=\; \tfrac12\log 2\pi S + \tfrac{(z_0 - m)^2}{2S},
\qquad
m = \mu_0 + \textstyle\sum_k u_k,
\quad
S = \Sigma_0 + \textstyle\sum_k Q_k + R
$$

$= 1.3280\ldots$ for the numbers above. **The graph is never told this.** It sums local
energies and local entropies and applies the counting correction, and the marginal likelihood
of the GPS reading is what is left. This is AutoBayes Remark 24 (see
[[Parameterized Statistical Game]]) surviving a generalisation the paper does not make: from
a two-factor composite to an $n$-factor graph, via [[Bethe Free Energy]].

The counting correction is doing real work here, and the chain is the cleanest place to see
why. Every variable has degree exactly $2$, so

$$
c_v \;=\; 1 - d_v \;=\; -1 \quad\text{for every } v,
\qquad
\sum_f 1 + \sum_v c_v \;=\; \chi(g) \;=\; 1
$$

Each of the two factors touching a variable charges $-H_v$ in its own `negentropy` summand,
so the shared entropy is counted twice; the correction adds one copy back. Drop it and $F$
is wrong by $\sum_v H_v$ — on a chain, wrong by an amount that *grows with $n$*, which is
what makes a chain a better test than the two-factor model of `gaussian.md` §4, where the
error is a single term and easy to mistake for a constant.

## 5. Turning the crank the other way

Unclamp the measurement and the same graph predicts instead of estimating. Delete the
`DataFactor` and `marg.z` becomes the *predictive* distribution
$\mathcal{N}(m, S) = \mathcal{N}(4, 2)$ — the pushforward of the prior along the whole chain,
noise accumulated at every step. Clamp $x_1$ instead of $z$ and it becomes dead reckoning.
Clamp both ends and it is a bridge.

None of these are separate code paths. They are the same graph with the clamps moved, and
[[Polarity Resolution]] works out which direction each factor must be run in from *which
messages have arrived* — a fact about the graph at that moment, not a fact about the model.
This is the property the README calls learning relations rather than functions, and the chain
is the smallest example where it is visible.

## 6. What this is not

> [!warning] This is a linear factor graph, not a SLAM system
> GTSAM's real value is in the layer *above* this one, and none of it is here.

- **No nonlinearity.** A `Pose2` chain has $x \ominus y$ in the residual, and a real library
  linearises at the current estimate and iterates (Gauss–Newton, Levenberg–Marquardt,
  Dogleg). This library has no relinearisation loop; `GaussianFactor` is linear by
  construction. Iterating BP on a *linearised* graph and iterating the linearisation point
  are different loops, and only the first exists here.
- **No manifolds.** Poses live on $SE(2)$/$SE(3)$; beliefs would need to be Gaussians in a
  tangent space with a retraction, and `combine` would need to agree about *which* tangent
  space. `GaussianBelief` is flat.
- **No loops.** A chain is a tree. Close the loop — the actual reason SLAM is hard — and
  `istree(g)` goes false, `tree_schedule` is no longer available, and everything in §3
  becomes approximate. See [[Loopy Message Passing]].
- **No ordering.** GTSAM's performance is variable ordering (COLAMD, nested dissection) and
  incremental updates (iSAM2). `Mycelium` has [[Schedules|message schedules]], which are the
  same information organised differently, and no elimination-order heuristics at all.
- **The odometry measurements are parameters.** $u_k$ sits in `ps` and
  `islearnable(::GaussianFactor)` is `true`, so a gradient step would happily *edit the
  odometry readings* to fit the GPS better. Nothing in the type system distinguishes "measured
  input" from "learnable weight". That is the honest state of the library, not a design
  position, and it is the gap [[Everything is a Factor]] is pointing at when it says data
  should be factors: $u_k$ ought to be a clamped variable on a third channel, not an entry
  in `ps`.

## 7. What this example caught

Two things, both of which passed on the two-factor model in `runtests.jl`:

1. **`combine` was ambiguous on `(TrivialBelief, GaussianBelief)`.** `messages.jl` had
   `combine(::TrivialBelief, b)` (untyped second argument) alongside the catch-all
   `combine(a::AbstractBelief, b::AbstractBelief)` that throws. Neither is more specific than
   the other, so Julia reported a `MethodError: ... is ambiguous` from inside
   `excluded_marginal` — i.e. the unit law for message pooling was unreachable for the one
   belief type that can actually be pooled. Fixed by restating the unit law at the
   `AbstractBelief` level. The failure was total rather than subtle only by luck: had the
   catch-all returned something instead of throwing, the ambiguity would have resolved
   silently in whichever order the methods were defined.

2. **The chain is where a missing counting correction stops looking like a constant.** See
   §4.

## Related

- [[Factor Graphs]] — the bipartite structure and `istree`
- [[Messages are Inversions]] — why running `odo1` backwards is exact and free
- [[Schedules]] — `tree_schedule`, and why two sweeps suffice
- [[Bethe Free Energy]] — the counting correction of §4
- [[Everything is a Factor]] — why the GPS reading needs its own variable
- [[Polarity Resolution]] — how §5 works without new code
- [[Loopy Message Passing]] — what breaks when the chain closes
