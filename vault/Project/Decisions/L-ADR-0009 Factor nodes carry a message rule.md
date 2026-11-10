#design

> **Status: accepted (2026-11-01, Daniel: "option 2 of ADR"; proposed the same day by Claude, implemented on `experiment-qm` and ported to `dev-danielb`).** Related: [[Extending Mycelium]], [[Switching Constraint Factors]] §6, [[Composing Stochastic Factors]] §3, [[passing]], [[L-ADR-0006 Separate likelihood_message from invert]].

## Context

Until now every factor node in Mycelium computed its messages one way: belief propagation. To send
on channel $t$, `step!` collects the cavity messages of the factor's other channels, resolves a
polarity and calls `factor_message`. That is exact on trees with closed-form beliefs, and it is what
every factor implements.

The experiments on `experiment-qm` found a case where that is the wrong rule for one node. In the
additive wiring $z = c + \sum_l k_l \delta_l + \varepsilon$, a sum node fed by blocks of switches:

- belief propagation at the sum node sees the other blocks only through their **prior** moments,
  since their mixture messages are collapsed before use, and learns parameters 0.30–0.44 nats from
  the truth;
- iterating with the other blocks' **current** means (structured mean field) reaches 0.14–0.16;
- exact enumeration reaches 0.03–0.04.

The fix needs the sum node to read its neighbours' full marginals, while the block factors keep
belief propagation, which is exact inside each block. That is the hybrid of Riegler et al. (2013):
factors split into a belief-propagation set and a mean-field set, with one free energy over both.

## Options

1. **A wrapper factor**, `MeanFieldFactor(f)`, forwarding every trait of `f` (channels, polarities,
   parameters, energies, free energies) and overriding the messages. It needs no change to the graph
   types, but a dozen forwarded methods, and every later rule would be another wrapper; wrappers of
   wrappers would be legal and meaningless.
2. **A message rule on the factor node**: `FactorNode` gains a field
   `rule::AbstractMessageRule`, set by `factor!(b, name, f; rule = …)`, with `BeliefPropagation()` as
   the default. `step!` dispatches on it, and each factor opts in to a rule by implementing that
   rule's message function (`mean_field_message` for `MeanField()`).
3. **A graph-level mode** ("run this graph by mean field"). Simple, but it cannot express the
   hybrid, which is the case that was needed.
4. **A rule per edge.** More general than per node, but the mean-field message of a factor is defined
   by the factor's expectation over *all* its other channels, so a per-edge rule has no clear meaning
   for one factor's messages.

## Decision

Option 2, accepted by Daniel. The rule is a property of how a node takes part in inference, not of the factor
(the same `LinearConstraintFactor` is a belief-propagation node in one graph and a mean-field node in
another), and not of the whole graph. It costs one field with a backwards-compatible default, and new
rules are new types plus opt-in methods, with no new wrappers.

Implemented:

- `AbstractMessageRule`, `BeliefPropagation`, `MeanField` (`graph.jl`);
- `mean_field_message(factor, target, polarity, beliefs, ps, st)`, with no generic method, so a
  factor without a rule fails with an `ArgumentError` naming the function to implement (`passing.jl`);
- `step!`: for a mean-field node, the other variables' full marginals replace the cavities, and
  everything else (polarity resolution, legality checks, damping, residuals) is unchanged;
- rules for `LinearConstraintFactor` and `SwitchingConstraintFactor` in Lenticulum (only means enter,
  since $\log f$ is quadratic in the continuous channels);
- **the graph free energy refuses** graphs with a node that is not on belief propagation:
  `_check_energy_edges` throws, so `bethe_free_energy`, `scalar_free_energy`, `free_energy_gradient`
  and `parameter_step` do too. The hybrid's free energy is not the Bethe free energy, and reporting
  Bethe's would be wrong.

  *Update (2026-11-02):* for `MeanField()` the refusal is replaced by the hybrid's own free energy,
  [[L-SPEC-005 The Hybrid Free Energy]], as the last consequence below anticipates ("each new rule
  must come with its free energy, or keep the refusal"). Paths that differentiate through
  inference still refuse mean-field nodes.

Tests: Mycelium's `test/message_rules.jl` (the default; that a mean-field node sees marginals, not
cavities; the refusals; the missing-method error) and Lenticulum's `lib/Lenticulum.jl/test/switching_constraint.jl` (the
two rules against their formulas and Monte Carlo; a hybrid graph against an independent
structured-mean-field implementation to $10^{-6}$).

## Consequences

- `FactorNode` has a fifth field. Its three- and four-argument constructors are kept and default to
  belief propagation, so existing code builds the same graphs.
- Graphs with mean-field nodes infer but cannot yet be trained by `parameter_step`; their M-step is
  done outside the graph. Lifting that is the first item of [[Extending Mycelium]] (the hybrid free
  energy).
- One tree sweep is not a fixed point of a hybrid graph. Callers iterate a `SequentialSchedule`; a
  schedule constructor that knows the rules is on the plan.
- The rule is the extension point for the rest of the plan: expectation propagation and mixture
  reduction become further rules with their own message function and free energy, rather than
  per-factor special cases. Each new rule must come with its free energy, or keep the refusal.
