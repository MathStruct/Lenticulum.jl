"""
    MyceliumLearning

Learning on a `Mycelium` factor graph: what changes the parameters, as opposed to what infers
the beliefs.

| file | what |
|---|---|
| `gradient.jl` | `free_energy_gradient` of a graph (the envelope gradient at fixed factor beliefs, LP-020) and `parameter_step` |
| `gradient_couplings.jl` | per-edge couplings: replay of a finite schedule, score estimates, `StraightThroughFactor` |
| `graph_adjoint.jl` | the implicit adjoint of a converged message fixed point |
| `em.jl` | closed-form M-steps from cluster beliefs: `em_step`, `fit_em`, the `m_step` hook |
| `sample_gradient.jl` | the gradient averaged over the samples of a `SampleBuffer` |

Inference (graphs, messages, schedules, plates, cluster nodes, the free energy itself) is in
`Mycelium`, which does not depend on this package. A factor type opts in to learning by
implementing `Mycelium.factor_belief` and `Mycelium.belief_free_energy` (for the gradient)
and `MyceliumLearning.m_step` (for EM).
"""
module MyceliumLearning

using Random: Random, AbstractRNG
using LuxCore: LuxCore
using LinearAlgebra: LinearAlgebra, logdet, tr
using LenticulumCore: LenticulumCore, AbstractGradientCoupling, DiagonalCoupling,
    GaussianBelief, dimension, isproper, logpartition
using Mycelium: Mycelium

# The names of Mycelium this package uses or adds methods to. Generated from the sources; add a
# name here when a new one is needed.
import Mycelium:
    Absorbing, AbstractMessageStore, BeliefPropagation, ClusterFactor, ClusterForm, DataFactor, 
    FactorGraph, FloodingSchedule, ForwardBackwardSchedule, FrozenClusterBelief, 
    FrozenClusterBeliefs, GibbsSampler, GradientDescent, MeanField, Message, MessageStore, 
    MessageTask, Momentum, Nesterov, OptimiserFactor, ResidualSchedule, SampleBuffer, 
    SequentialSchedule, TreeSchedule, _EnergyRow, _active_indices, _beliefs_scalar, 
    _check_energy_edges, _empty_graph_state, _energy_rows, _factor_nodes, _factor_parameters, 
    _gather_active, _indexed, _inputs, _key_groups, _key_node, _mask_gradient, _polarity, _row_messages, _setsubtree, 
    _stored, _subtree, _switch_columns, _variable_nodes, _variable_term, all_tasks, as_cluster, 
    available_channels, belief_distance, belief_free_energy, belief_logdensity, can_absorb, 
    can_emit, check_legal, cluster, cluster_form, damp, edges_of_factor, edges_of_variable, emits, 
    excluded_marginal, factor_belief, factor_beliefs, factor_free_energies, factor_message, 
    factornode, infer!, islegal, issink, learns_unnormalised, leaves, local_free_energy, marginal, 
    marginals_into, mean_field_energy, messages_into, nedges, parameter_keys, point, propagate!, 
    replica, replicas, resolve_polarity, rule_get, rule_put, sample_weights, step!, sweep!, tasks, 
    tied_factors, transfer, uses_cavity, variable

include("gradient.jl")
include("gradient_couplings.jl")
include("graph_adjoint.jl")
include("sample_gradient.jl")
include("em.jl")

export parameter_step, GradientEstimate, StraightThroughFactor, sample_gradient
export em_step, fit_em, m_step, m_step_signature, moment_statistics, MomentStats

end # module
