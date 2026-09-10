# Metabolomics and Proteomics

> Where the exact relation is **stoichiometry**, the system is famously underdetermined, and a
> posterior therefore says something a point estimate structurally cannot.

## The relations

At metabolic steady state, the stoichiometric matrix ``S`` and the flux vector ``v`` satisfy

$$S\,v \;=\; 0$$

— one linear constraint per metabolite, saying what is produced equals what is consumed. That is
a `LinearConstraintFactor` per metabolite, with the stoichiometric coefficients as the
``A_i``, and it is exactly the fragment where this project's results are exact.

Around it sit relations of decreasing certainty: enzyme kinetics (Michaelis–Menten, partly
known), thermodynamic constraints on directionality (known in form, uncertain in parameters),
and regulation (largely unknown).

## Underdetermination is the whole problem, and it is an argument for posteriors

``Sv = 0`` has far more fluxes than metabolites, so the feasible set is a high-dimensional
polytope, not a point. Flux Balance Analysis handles this by **choosing an objective** —
maximise growth, typically — and returning one vertex.

That is a modelling decision disguised as a computation, and it is where a Bayesian treatment
earns its place:

> FBA returns one flux vector from an underdetermined system by picking an objective.
> A posterior returns **the feasible set with weights** — which is what the data actually
> supports.

Where measurements pin fluxes down, the posterior is narrow; where they do not, it is wide, and
the width is the honest statement of what is known. Reporting a single FBA solution for an
unconstrained subnetwork is reporting the objective, not the biology.

## What is observed

Metabolite concentrations (LC-MS, sparse and semi-quantitative), ``^{13}\mathrm{C}`` isotope
labelling patterns, protein abundances (proteomics, different instrument, different noise
model, different coverage), and exchange fluxes at the boundary. Different modalities, wildly
different reliabilities, and coverage in the low tens of percent.

**Missing data is the normal case**, which is what `Observed` / `Unobserved` / `Latent`
encodes natively rather than by imputation.

## What would be learned

- **Kinetics** where the mechanism is unknown — a fitted flux–concentration relation.
- **Regulatory couplings**, which is mostly what is not in the stoichiometry.
- **Instrument response models** relating true abundance to measured intensity — genuinely
  learnable and genuinely uncertain.

Exact stoichiometry beside fitted kinetics is the grey-box case, and here it is not optional:
nobody has a mechanistic kinetic model of a whole metabolic network, and nobody would accept a
purely learned model that violated mass balance.

## What a residual means

An unmodelled reaction, a mis-annotated pathway, or a measurement error — and per-factor
attribution says which metabolite fails to balance. In a curated genome-scale model, a
persistently violated balance is evidence of a missing edge in the reconstruction, which is a
publishable result rather than a nuisance.

## One model, several questions

- Fluxes from concentrations and labelling.
- Concentrations predicted from a hypothesised flux state.
- **Which measurement would most reduce the uncertainty?** — an experimental-design question
  answered by asking what each candidate observation does to the posterior.

That last one is where uncertainty pays for itself in a field where each measurement is
expensive.

## What would be hard

- **Scale.** Genome-scale reconstructions have ``10^3``–``10^4`` reactions. Feasible in
  principle for a sparse linear system, not for the current implementation
  ([[Parallelism and Compilation]]).
- **The polytope is not Gaussian.** Fluxes have sign constraints and bounds, so the posterior is
  a truncated distribution on a polytope. Gaussian beliefs are the wrong shape, and sampling on
  a polytope is its own literature.
- **Cycles.** Metabolic networks are dense with cycles — the TCA cycle is the textbook example —
  so the graph is loopy from the start ([[Loopy Message Passing]]).
- **Steady state is an assumption**, and the interesting biology is often the transient. That is
  the [[Time as a Base]] extension, with all its caveats.

Related: [[Motivating Examples]], [[Channels and Polarity]], [[Loopy Message Passing]],
[[Time as a Base]], [[Parallelism and Compilation]]
