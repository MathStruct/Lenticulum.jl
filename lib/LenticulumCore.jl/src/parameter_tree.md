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
`copyto!` during that flatten would break Zygote. GPU support for this nested solve path
has not been tested. Core tree gradients and VD regression tests exercise the shared code.
