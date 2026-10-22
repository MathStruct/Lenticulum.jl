#design

> **Status: accepted (recorded 2026-10-22; in force since the project began).** Related: [[Why Julia]], [[Parallelism and Compilation]].

## Context

Lenticulum's workload is many small, heterogeneous factors, solvers inside differentiable programs, and query directions chosen at run time. A framework that traces a computation graph (PyTorch, JAX) handles that badly: graph breaks, recompilation for every new shape of query, and custom factors written in a second language.

## Options

1. Julia.
2. An extension of PyTorch or JAX.
3. C++, Rust or Mojo.

## Decision

Julia. In Julia ordinary code is the compiled, differentiable computation, so a user's factor, belief type or solver loop needs no special treatment. [[Why Julia]] gives the full comparison and the costs.

## Consequences

- Lux.jl's layer interface is the template for factors ([[L-SPEC-001 The Factor Interface]]).
- RxInfer.jl and ModelingToolkit.jl are neighbours to interoperate with rather than reimplement ([[RxInfer as a Backend]], [[ModelingToolkit as an Acausal Relation]]).
- The costs listed in [[Why Julia]] §6 are accepted: compile latency, uneven AD across backends, type stability as the programmer's job, and a smaller community.
