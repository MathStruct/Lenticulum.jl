#overview

> Specifications of Lenticulum.jl's interfaces, numbered `L-SPEC-nnn`. A specification states a contract precisely enough for tests and new code to cite it by section. It starts as a `draft`, describing the code as it is, and becomes `accepted` when Daniel accepts it; after that, changing it needs a decision ([[Decisions]]). Use the specification template (`vault/templates/Specification Template.md`).

| # | Specification | Status |
|---|---|---|
| [[L-SPEC-001 The Factor Interface]] | What a factor type implements, alone and in a graph: parameters, channels and polarities, the two energy levels, free energy, messages | draft |
| [[L-SPEC-002 Stop Gradient]] | Which stop-gradients exist (detach, freeze, coupling, one-way messages), how they are asked for, what the backward pass computes | planned |
| [[L-SPEC-003 Exposed Parameters and Energy Channels]] | Parameters as channels (outputs into weights, weight tying), energy taps (energies into inputs), energy beliefs and risk scalarisations | planned |
| [[L-SPEC-004 Algebraic Factors Toric and Grassmannian]] | The first code of the algebraic family: toric (log-linear) factors over discrete channels, Grassmannian parameters with retraction, then the polynomial factor and multi-branch roots; Plücker coordinates | partly implemented (§0, §1.1–§1.7, §2.1–§2.3 and §5; toric fitting, membership, Grassmannian updates, Sampson gauge obligation and forward Plücker transport checked; §2.4, §3–§4 planned) |
