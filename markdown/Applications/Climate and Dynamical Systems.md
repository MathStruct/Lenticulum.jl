# Climate and Dynamical Systems

> Data assimilation is factor-graph inference with different vocabulary, an enormous state, and
> a sixty-year head start. The interesting question is not whether the shape fits — it plainly
> does — but what a relational, grey-box treatment would add.

## The relations

| relation | says |
|---|---|
| the discretised dynamics | consecutive states are related by a PDE solve |
| conservation laws | mass, energy, momentum are preserved exactly |
| observation operators | a satellite radiance is a functional of the state |
| boundary and forcing | the system is driven by things it does not determine |

The conservation laws are the acausal ones and they are not negotiable: a scheme that violates
mass balance is wrong regardless of how well it fits. That is precisely a hard constraint
alongside soft ones — a `LinearConstraintFactor` with small noise beside factors with large
noise.

## The field already does this, under other names

**4D-Var** minimises a sum of a background term and observation terms over a time window. That
is minimising ``\sum_c E_c`` over a chain — [[Energy-Based Factor Graphs]] §1 exactly, and at
zero temperature (a MAP estimate, no posterior).

**The Ensemble Kalman Filter** propagates an ensemble and updates it against observations: a
sample-based belief with a Gaussian update rule, which is a particular answer to the `combine`
problem [[messages]] §1 records as open.

So both dominant methods are instances of the framework, and the framework's contribution would
have to be at the margins the existing methods leave: a genuine posterior rather than a MAP
(4D-Var), non-Gaussian updates (EnKF), and learned components inside the model rather than
bolted beside it.

## What is observed

Satellites, radiosondes, buoys, ships, stations — different instruments, different error
characteristics, wildly non-uniform coverage in space and time, and **asynchronous by nature**.
The observation operator is often the hardest part of the model.

Asynchrony over a continuum is the [[Time as a Base]] case: state as a trajectory queried at
observation times rather than a value at grid times.

## What would be learned

This is where the domain has moved and where the fit is sharpest. **Subgrid parametrisations** —
clouds, convection, turbulence, radiation — are the known unknowns of climate modelling, they
are the dominant source of inter-model spread, and they are increasingly being replaced by
learned emulators.

A learned convection scheme inside a model that must still conserve mass and energy is exactly
the grey-box case, and the failure mode of naive ML parametrisations is exactly the one a hard
constraint prevents: they drift, because nothing forces conservation.

> Learned components that must respect exact conservation laws is not a niche requirement here.
> It is the central open problem of ML-augmented climate modelling.

## What a residual means

**Model misfit, localised.** Per-factor energies say *where and when* the model disagrees with
observation, which is how parametrisation deficiencies are diagnosed. And the free energy is a
model-comparison score: which of two parametrisation schemes explains the observations better,
with complexity accounted for rather than penalised by hand.

For a field whose central output is an ensemble of structurally different models, a principled
comparison score is not a small thing.

## One model, several questions

- **Assimilation**: observations clamped, state inferred.
- **Prediction**: state clamped, future propagated.
- **Attribution**: clamp a counterfactual forcing and ask what the state would have been.
- **Observing-system design**: which new instrument most reduces posterior uncertainty?

The third is the socially consequential one and it is a polarity choice.

## What would be hard

This is the least tractable of the six and it should be said plainly.

- **Scale.** ``10^8``–``10^9`` state variables. Nothing in this project is within several orders
  of magnitude ([[Parallelism and Compilation]]), and operational assimilation is already at the
  limit of the largest machines available.
- **Chaos.** Sensitivity to initial conditions bounds predictability regardless of method, and
  linearised error propagation is valid only over short windows — which is why 4D-Var uses a
  window at all.
- **Strong nonlinearity**, so the exact linear-Gaussian results do not apply.
- **The dynamics factor is a PDE solve.** Wrapping one as a factor means each message costs a
  model run, which changes the economics of message passing entirely: you cannot iterate.

The realistic reading is that this domain is a **source of structure to learn from** rather than
a near-term target — 4D-Var and EnKF are worth studying as two well-tested answers to questions
this project has open, especially the EnKF's treatment of sample-based beliefs.

Related: [[Motivating Examples]], [[Energy-Based Factor Graphs]], [[Time as a Base]],
[[messages]], [[Parallelism and Compilation]], [[The Linear Gaussian Chain]]
