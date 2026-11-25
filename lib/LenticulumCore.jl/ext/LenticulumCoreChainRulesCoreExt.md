#implementation

> The ChainRules extension registers metadata queries and zero-buffer construction as discrete operations.
>
> Sources: code: `LenticulumCoreChainRulesCoreExt.jl`, `../src/gaussian_belief.jl`

## Element types

The inputs to `_compute_eltype` and `_zeros_like` determine the numeric type and array family.
The results are independent of the numerical values of those inputs, so these helpers have
zero tangents under reverse-mode AD.
