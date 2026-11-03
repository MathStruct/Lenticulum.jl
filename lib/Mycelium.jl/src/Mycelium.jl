"""
    Mycelium

Factor graphs and message passing for Lenticulum.jl.

`LenticulumCore` says what a **factor** is (a parameterized statistical game); `Mycelium`
says how factors are **wired** and in what **order** they talk. It is the layer that has no
counterpart in Lux, because in Lux the wiring is a DAG and the order is implied by it.

## The design in five sentences

1. The graph is **bipartite**: variables are wires, factors are nodes. Everything with
   content is a factor — data, priors, losses and optimisers included, following AutoBayes
   Remark 24. See `factors.jl`.
2. Edges are **directed**, and carry one of `Emitting`, `Absorbing`, `Bidirectional`. A
   bidirectional edge is exactly what "implicit" means.
3. A **factor → variable message is an inversion** ``c'_\\pi``: the target channel becomes
   `Unobserved()`, the channels with incoming messages become `Observed()`, the rest
   `Latent()`, and `assemble` + `invert` do the rest. See `polarity_resolution.jl`.
4. **Scheduling** is the whole problem. On a tree, two sweeps are exact; off a tree you are
   running loopy BP and nothing is guaranteed. See `schedules.jl`.
5. The **free energy** of a graph is the Bethe form: energies add, entropies get a counting
   correction ``(1-d_v)``. That is the same energy/entropy asymmetry AutoBayes identifies,
   one level up. See `free_energy.jl`.

## Two acyclicity notions, not one

| predicate | about | decides |
|---|---|---|
| `istree(g)` | the **undirected** bipartite graph | whether message passing is **exact** |
| `isdag(g)` | the **directed** multigraph | whether the graph is **Lux-compatible** |

They are independent, and confusing them is the commonest way to be wrong about a factor
graph.

Concept notes are in `vault/Factor Graphs/`; per-file implementation notes sit next to each
source file.
"""
module Mycelium

using DispatchDoctor: @stable
using Random: Random, AbstractRNG
using LuxCore: LuxCore
using LenticulumCore: LenticulumCore, AbstractGradientCoupling, DiagonalCoupling,
    GaussianBelief, dimension, isproper, logpartition
using LinearAlgebra: LinearAlgebra, logdet, tr

include("graph.jl")
include("polarity_resolution.jl")
include("blocks.jl")
include("messages.jl")
include("schedules.jl")
include("passing.jl")
include("plates.jl")
include("free_energy.jl")
include("factors.jl")
include("exposure.jl")
include("gaussian.jl")
include("gradient_couplings.jl")
include("graph_adjoint.jl")
include("sampling.jl")

# --- Graph -----------------------------------------------------------------
export FactorGraph, GraphBuilder, VariableNode, FactorNode, Edge
export AbstractMessageRule, BeliefPropagation, MeanField
export variable!, factor!, connect!, build, validate
export EdgeDirection, Bidirectional, Emitting, Absorbing, emits, absorbs
export nvariables, nfactors, nedges, variable, factornode
export edges_of_factor, edges_of_variable, variable_degree, factor_degree, channels_of
export euler_characteristic, isconnected, istree, isdag, topological_order, leaves
export learnable_factors, parameter_key, parameter_keys, tied_factors
export Plate, PlateBuilder, plate!, plates, replica_name, PlateSchedule, plate_schedule

# --- Polarity --------------------------------------------------------------
export PolarityError, resolve_polarity, check_legal, can_emit, can_absorb
export legal_targets, legal_sources

# --- Messages --------------------------------------------------------------
export Message, MessageStore, reset!, pin!, unpin!, combine, combine_with_overlap, belief_logdensity, belief_distance
export marginal, excluded_marginal, damp, can_damp
export has_to_variable, has_to_factor
export BeliefBlock, TrivialBlock, PointBlock, GaussianBlock, VectorBlock, belief_block, replica, replicas, nreplicas

# --- Schedules -------------------------------------------------------------
export AbstractSchedule, MessageTask, tasks, all_tasks, islegal
export SequentialSchedule, FloodingSchedule, TreeSchedule, ForwardBackwardSchedule,
    ResidualSchedule
export flooding_schedule, tree_schedule, forward_backward_schedule, HybridSchedule, hybrid_schedule

# --- Passing ---------------------------------------------------------------
export factor_message, factor_messages, mean_field_message, mean_field_energy, available_channels, step!, sweep!, propagate!, infer!
export ConvergenceReport

# --- Sampling --------------------------------------------------------------
export sample_belief, GibbsSampler, GibbsState, gibbs_init, gibbs_sweep!, gibbs_points
export factor_residual_moments, AbstractSampleFilter, WindowFilter, DecayFilter
export SampleBuffer, record!, advance!, sample_weights, energy_moments, free_energy_mean
export residual_moments, sample_gradient

# --- Free energy -----------------------------------------------------------
export counting_number, bp_degree, counting_numbers, total_counting_number, variable_entropy
export factor_free_energies, variable_corrections, messages_into, variable_free_energy
export bethe_free_energy, chain_free_energy   # scalar_free_energy extends LenticulumCore's

# --- Factors ---------------------------------------------------------------
export local_free_energy, local_free_energies, factor_beliefs, belief_free_energies, marginals_into, uses_cavity, issink, point, factor_belief, belief_free_energy
export DataFactor, PriorFactor, LossFactor, RelayFactor, OptimiserFactor
export ExposedFactor, expose, supports_parameter_polarity, parameter_message, parameter_free_energy
export AbstractUpdateRule, GradientDescent, Momentum, Nesterov
export rule_get, rule_put, optimiser_step, parameter_step
export GradientEstimate, StraightThroughFactor

end # module
