#implementation

> Numeric Lux parameter trees share one flatten/rebuild implementation between core
> derivatives, ImplicitLayers and VariationalDiffusion.

> Sources: original implementation; code: `parameter_tree.jl`, `../test/parameter_ad.jl`, `../../VariationalDiffusion.jl/ext/VariationalDiffusionDifferentiationInterfaceExt.jl`
>
> Theory (CT-ML wiki): [Parametric Lens](https://mathstruct.org/CategoryTheory-ML-Wiki/Parametric-Lens)

Tuple/NamedTuple fields are traversed depth first, arrays in column-major order.
Offsets and shapes are computed outside the differentiated function. Rebuilding only
views and reshapes the active vector, preserving its element type (including duals),
container keys, scalar leaves and empty subtrees. Gradients use the same reconstruction.

These are internal helpers, not a general Functors/ComponentArrays protocol. Leaves must
be numeric scalars or arrays, with tuple/NamedTuple containers. Mixed numeric types
promote to one vector element type; empty containers are neutral, so they cannot
widen Float32 parameters to Float64. The core tests check this case. `_flatten` allocates alongside the first array leaf,
preserving VariationalDiffusion's device behaviour. `_flatten_active` uses functional
concatenation when a DEQ solve is nested inside an outer reverse-mode tape. Array
leaves are copied before concatenation so rebuilt views do not introduce array/view
unions into Enzyme's type analysis; see `ImplicitLayers.jl/test/parameter_ad.jl`. Mutating
`copyto!` during that flatten would break Zygote. The JLArrays subset of this nested solve path is tested below. Core tree gradients and VD regression tests exercise the shared code.

## LP-011 implementation (2026-10-27)

`_freeze_paths` normalises one path, several paths, or `:all`;
`_trainable_indices` validates every selected subtree and computes its active numeric
coordinates in the existing flatten order. Optimiser state allocates only these
coordinates. `stop_gradient` is identity on primal values with optional ForwardDiff
and ChainRules zero-derivative rules. Shape/index computations are nondifferentiable.

Evidence: `lib/Mycelium.jl/test/gradient_couplings.jl` and
`test/gradient_couplings.jl`. API, mixed Jacobians, costs and unsupported combinations
are specified in [[gradient_couplings]] and [[L-SPEC-002 Stop Gradient]].


## Device empty branches (LP-002 batch B, 2026-10-27)

Functional flattening dispatches through the first array leaf's storage. CPU
concatenation keeps its prior arithmetic. The GPUArraysCore extension removes
empty concatenation parts: an empty tuple otherwise introduces host `Float64[]`,
selects a host `vcat` and scalar-indexes the active device array. An all-empty
device tree retains a copied empty device leaf and its element type. Array views
from parameter reconstruction use the same device path. No active numeric value
is inspected to select the storage behavior.

`lib/ImplicitLayers.jl/test/device_arrays.jl` checks nested empty branches and
all-empty device arrays in Float32/Float64, including solved-state parameter
gradients under ForwardDiff/Zygote against dense closed forms. Scalar parameter
leaves remain unsupported on device vectors; use one-element arrays. The final
full Core and dependent package runs verify the shared CPU helper too.
