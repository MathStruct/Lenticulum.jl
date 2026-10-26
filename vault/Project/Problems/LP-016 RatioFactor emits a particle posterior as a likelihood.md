#open-problem

> **Status: open (2026-10-26).** Related: [[LP-008 No conformance test for the factor interface]], [[LP-009 Game-level free energy is declared but not implemented]], [[L-SPEC-001 The Factor Interface]], [[ratio]].

> Sources: original repository analysis; code: `lib/Adversarial.jl/src/ratio.jl`, `lib/Adversarial.jl/test/conformance.jl`, `lib/Mycelium.jl/src/messages.jl`
>
> Theory (CT-ML wiki): [Bayesian Inversion](https://mathstruct.org/CategoryTheory-ML-Wiki/Bayesian-Inversion) · [Statistical Game](https://mathstruct.org/CategoryTheory-ML-Wiki/Statistical-Game)

**Problem.** RatioFactor's particle inversion and outgoing message both reweight the cavity.
With p=N(0,1), q=N(1,1), log(p/q)=1/2-x and incoming particles {0,1} weighted {1/2,1/2},
the emitted weights are {e/(1+e),1/(1+e)}. This is the posterior. The generic
`likelihood_message` returns it too. Pooling it with the cavity fails because two
SampleBeliefs have no density product rule. Moreover `factor_message` reads a supplied
target-channel input and lets it override the cavity, violating target exclusion.

The graph objective averages -log(p/q) under the incoming cloud (zero here); the posterior
expectation is 1/(1+e)-1/2, which differs. Neither equality of these expectations nor a
posterior entropy can be assumed when adding a Definition 20 game API.

**Options.** Supply an explicit likelihood/density-ratio belief with a valid pooling rule and
target exclusion; or restrict RatioFactor to a documented posterior-reweighting operation
outside extrinsic BP. A joint game objective would also need an entropy/reference-measure
choice. Silently pooling the posterior again would count the cavity twice.

**Resolution so far.** The point-prior case has a tested game API and passes the common
checker. `lib/Adversarial.jl/test/conformance.jl` independently checks the two-particle
weights and retains three `@test_broken` assertions for pooling, exclusion and the
expectation mismatch. Non-Dirac game priors throw `ArgumentError`. The particle defect
is left open; it is not a small arithmetic fix.
