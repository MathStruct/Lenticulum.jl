# Energy Markets and Power Grids

> The domain where the physics and the economics are **both** networks of constraints, and
> where the standard method is already a factor graph under another name.

## The relations

Two layers, both acausal, sharing variables:

**Physical.** Kirchhoff's current law at every bus (``\sum_i i_i = 0``), Kirchhoff's voltage law
around every loop, line-flow equations relating flows to voltage angles, and the generation–load
balance. None has an input or an output — a bus does not "cause" its incident currents.

**Economic.** Market clearing, bid stacks, ramp limits, transmission constraints, and the
locational prices that come out of them. Also relations rather than assignments.

The `LinearConstraintFactor` example in this project is a three-way Kirchhoff node, and the DC
power-flow approximation is linear — so an entire useful class of grid model lands in the exact
fragment.

## The standard method is already this

Power-system **state estimation** is weighted least squares over a network of measurement
residuals, run continuously on every transmission grid in the world. That is a Gaussian factor
graph with a MAP estimator, built before anyone called it one.

Which means the framework does not have to argue its way in — it has to justify what it adds:
a posterior instead of a point, learned components alongside the physics, and one model
answering more than one question.

## What is observed

SCADA measurements at some buses, PMUs at a few, meter readings at slow cadence, and nothing at
all at most distribution-level nodes. Partial observation is structural — grids are
*chronically* under-instrumented at the edges — and the observability analysis that grid
operators run is precisely the question of whether a polarity is well posed.

## What would be learned

- **Renewable generation forecasts** — wind and solar, the dominant source of uncertainty.
- **Demand models**, increasingly with behind-the-meter solar and storage that nobody measures.
- **Distribution-network topology and impedances**, which are frequently wrong in the records.

All beside exact Kirchhoff constraints. That is the grey-box case in its clearest industrial
form: physics you would never want to learn, next to quantities you can only learn.

## What a residual means

**Bad data.** Residual analysis is the standard method for detecting failed sensors, and it is
the same computation as the graded energy. Per-factor attribution answers "which measurement
disagrees with the network", which is what an operator acts on.

It also detects **topology errors** — a breaker whose recorded state is wrong makes a whole
region of the graph inconsistent, and that shows up as a spatially structured pattern of
residuals rather than a single outlier.

## One model, several questions

- **State estimation**: measurements clamped, state inferred.
- **Contingency analysis**: hypothesise a line outage, propagate, ask what the state would be.
- **Observability**: is this set of measurements enough to determine the state? — a structural
  question about the graph, not a numerical one.
- **Optimal dispatch**: clamp costs and limits, solve for generation.

Same network, different clamps. [[Channels and Polarity]].

## What would be hard

- **AC power flow is nonlinear** — the exact results here are linear-Gaussian, so full AC is
  approximate. DC is linear and widely used, which is a real foothold, but it is an
  approximation before you start.
- **Meshed grids are loopy**, and Kirchhoff's voltage law is a statement *about loops*. So the
  loopy-message-passing problem is not incidental to this domain, it is intrinsic:
  exact means, wrong variances ([[Loopy Message Passing]]) — and for a risk-constrained
  dispatch decision the variance is the point.
- **Scale**: ``10^4``–``10^5`` buses for a transmission network, more with distribution. See
  [[Parallelism and Compilation]].
- The bootstrap problem is real here too — a loop of pure constraints with no priors cannot
  start propagating (`constraint.md` §4.2), and a grid model is exactly such a loop.

Related: [[Motivating Examples]], [[ModelingToolkit as an Acausal Relation]],
[[Channels and Polarity]], [[Loopy Message Passing]], [[Parallelism and Compilation]]
