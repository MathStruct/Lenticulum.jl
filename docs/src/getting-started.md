# [Getting started](@id getting-started)

A complete, runnable example: estimating a robot's trajectory from odometry and one GPS
reading. This is the problem every factor-graph library opens with, and it exercises most of
the framework in about thirty lines.

Every code block on this page is executed when the documentation is built, so the numbers
below are real output.

## The problem

A robot moves along a line. We have:

- a **prior** on where it started: ``x_1 \sim \mathcal{N}(0, 1)``
- two **odometry** readings, each saying "you moved 2 units, give or take":
  ``x_{k+1} = x_k + 2 + \varepsilon``
- one **GPS** reading of the final pose: ``z = x_3 + \delta``, measured at ``4.5``

Odometry alone would put the robot at ``0, 2, 4``. The GPS says ``4.5``. The right answer is
neither — it is whatever weighs the two by their noise.

## Building the graph

```@example chain
using Lenticulum, LenticulumCore, Mycelium
using LenticulumCore: DiracBelief

μ₀, Σ₀ = [0.0], fill(1.0, 1, 1)              # prior on x₁
u      = ([2.0], [2.0])                       # odometry readings
Q      = (fill(0.25, 1, 1), fill(0.25, 1, 1)) # odometry noise
R      = fill(0.5, 1, 1)                      # GPS noise
z₀     = [4.5]                                # the GPS reading
Id     = fill(1.0, 1, 1)

b = GraphBuilder()

# variables: three poses, plus one auxiliary variable for the measurement
for v in (:x1, :x2, :x3, :z)
    variable!(b, v, 1)
end

# factors
factor!(b, :prior, GaussianPrior(:x1, μ₀, Σ₀))
factor!(b, :odo1, GaussianFactor(1 => 1; noise = Q[1], channels = (:x, :y)))
factor!(b, :odo2, GaussianFactor(1 => 1; noise = Q[2], channels = (:x, :y)))
factor!(b, :gps,  GaussianFactor(1 => 1; noise = R,    channels = (:x, :y)))
factor!(b, :data, DataFactor(:z, z₀))

# wiring: which factor channel attaches to which variable
connect!(b, :prior, :x1, :x1; direction = Bidirectional())
connect!(b, :odo1, :x, :x1; direction = Bidirectional())
connect!(b, :odo1, :y, :x2; direction = Bidirectional())
connect!(b, :odo2, :x, :x2; direction = Bidirectional())
connect!(b, :odo2, :y, :x3; direction = Bidirectional())
connect!(b, :gps,  :x, :x3; direction = Bidirectional())
connect!(b, :gps,  :y, :z;  direction = Bidirectional())
connect!(b, :data, :z, :z;  direction = Emitting())

g = validate(build(b))
```

Three things worth noticing in that block:

- **An odometry factor is just a `GaussianFactor` with ``A = I``.** ``x_{k+1} = x_k + u_k``
  is ``y = Ax + b`` with ``A = I`` and ``b = u_k``. There is no special "between factor" type;
  the offset lives in `ps`, below.
- **The GPS reading needs its own variable.** A measurement is evidence, and evidence is a
  factor — so the measurement becomes a `GaussianFactor` (the noise model) plus a
  `DataFactor` (the reading itself) on an auxiliary variable `:z`. That separation lets you
  swap the reading without touching the noise model.
- **Edge directions.** `Bidirectional()` means messages flow both ways along that edge;
  `Emitting()` means the factor only ever speaks. The data clamp only emits.

## Parameters and inference

Parameters are separate from the graph, exactly as in Lux:

```@example chain
ps = (prior = NamedTuple(),
      odo1 = (A = Id, b = u[1]),
      odo2 = (A = Id, b = u[2]),
      gps  = (A = Id, b = [0.0]),
      data = NamedTuple())

st = (prior = NamedTuple(), odo1 = NamedTuple(), odo2 = NamedTuple(),
      gps = NamedTuple(), data = NamedTuple())

istree(g)     # a chain with things hanging off it is a tree
```

Because it is a tree, `tree_schedule` gives an **exact** answer in two sweeps:

```@example chain
marg, report, st, store = infer!(g, tree_schedule(g), ps, st)
report.converged
```

## Reading the answer

```@example chain
for v in (:x1, :x2, :x3)
    println(v, ":  mean = ", round(belief_mean(marg[v])[1]; digits = 4),
               "   variance = ", round(belief_cov(marg[v])[1]; digits = 4))
end
```

Two things to look at.

**The poses are pulled toward the GPS reading**, but not all the way — the odometry gets a
say, weighted by its noise.

**The variance of ``x_1`` is below its prior of 1.0.** Nothing told ``x_1`` about the GPS
directly; the information travelled backwards down the chain, from `z` to `x3` to `x2` to
`x1`. That requires each odometry factor to be run in the *opposite* direction to the one it
was written in — which is the entire point of a factor being a relation rather than a
function. A filter would report 1.0 here; this is a smoother, and it came free from the
schedule.

## The evidence, for nothing extra

The graph also knows how surprising the GPS reading was:

```@example chain
total, st = LenticulumCore.scalar_free_energy(store, g, ps, st)
total
```

That number is ``-\log p(z_0)``, the negative log marginal likelihood of the measurement. It
is never computed directly — the graph sums local energies and entropies with a correction
term, and the marginal likelihood is what is left. Check it against the closed form:

```@example chain
m = μ₀ .+ u[1] .+ u[2]                    # where the prior says x₃ should be
S = Σ₀ .+ Q[1] .+ Q[2] .+ R               # with this much accumulated variance
d = z₀ .- m
(log(2π) + log(S[1]) + (d' * (S \ d))[1]) / 2
```

Being able to score a model without being told how is what makes the free energy worth
carrying around.

## Running the same factor the other way

Nothing about `:gps` says it is a measurement. Delete the `DataFactor` and the same graph
*predicts* the reading instead of conditioning on it:

```@example chain
b2 = GraphBuilder()
for v in (:x1, :x2, :x3, :z)
    variable!(b2, v, 1)
end
factor!(b2, :prior, GaussianPrior(:x1, μ₀, Σ₀))
factor!(b2, :odo1, GaussianFactor(1 => 1; noise = Q[1], channels = (:x, :y)))
factor!(b2, :odo2, GaussianFactor(1 => 1; noise = Q[2], channels = (:x, :y)))
factor!(b2, :gps,  GaussianFactor(1 => 1; noise = R,    channels = (:x, :y)))
connect!(b2, :prior, :x1, :x1; direction = Bidirectional())
connect!(b2, :odo1, :x, :x1; direction = Bidirectional())
connect!(b2, :odo1, :y, :x2; direction = Bidirectional())
connect!(b2, :odo2, :x, :x2; direction = Bidirectional())
connect!(b2, :odo2, :y, :x3; direction = Bidirectional())
connect!(b2, :gps,  :x, :x3; direction = Bidirectional())
connect!(b2, :gps,  :y, :z;  direction = Bidirectional())
g2 = validate(build(b2))

ps2 = (prior = NamedTuple(), odo1 = (A = Id, b = u[1]),
       odo2 = (A = Id, b = u[2]), gps = (A = Id, b = [0.0]))
st2 = (prior = NamedTuple(), odo1 = NamedTuple(),
       odo2 = NamedTuple(), gps = NamedTuple())

pred, _, _, _ = infer!(g2, tree_schedule(g2), ps2, st2)
belief_mean(pred.z), belief_cov(pred.z)
```

The predictive distribution: mean 4, variance 2 — the prior pushed along the whole chain with
noise accumulating at every step. Clamp `:x1` instead and it becomes dead reckoning; clamp
both ends and it is a bridge. Same graph, clamps moved.

## What to read next

- **[Vocabulary](@ref vocabulary)** if any of the words above were doing unexplained work.
- The [package pages](@ref lenticulumcore) for the reference documentation.
- `markdown/Mycelium/The Linear Gaussian Chain.md` in the vault for this example worked
  through properly — why it is exact, what the counting correction is doing, and what this
  library is *not* (no nonlinearity, no manifolds, no loop closure).
