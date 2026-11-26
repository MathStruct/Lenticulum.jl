#implementation

> Tied cluster replicas share the precision classes and the stacked solves described in
> `cluster.md`. Their joint arrays, output blocks, and frozen beliefs follow the incoming
> storage backend.

> Sources: original to this repository, code: `lib/Mycelium.jl/src/cluster_batch.jl`, tests:
> `lib/Lenticulum.jl/test/device_inference.jl` and `lib/Lenticulum.jl/test/device_graphs.jl`
> Theory (CT-ML wiki): see `cluster.md`

The first replica establishes the joint's storage and precision classes. Point columns,
information vectors, Cholesky factors and residual products use that storage across the
stacked solve. Class and assignment indices are structural host data. A batch whose
replicas have different input structure follows the per-replica rule. The frozen plate's
numeric arrays follow the device, and the final scalar parts of its graded energies are
transferred in batches to build the host collection.
