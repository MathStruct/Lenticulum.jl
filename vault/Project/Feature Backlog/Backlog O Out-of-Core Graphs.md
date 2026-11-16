#overview

> Section O of the [[Feature Backlog]]: a graph kept in a store and loaded where a query reaches; planned as its own package. Open rows only; a row that is
> done moves to [[Backlog Done 2026-11]] with its id. Status, priority, ids and the column *needs*
> are explained in [[Feature Backlog]], which also holds the dependency graph.

> Sources: original to this repository; the sources of each row are in its *source* column.

A graph too big for memory, kept in a store and loaded where a query reaches. It is planned as its
own package, `lib/MyceliumStore.jl` (working name), on top of Mycelium. Daniel, 2026-11-05: not only
for language models but also for heterogeneous multi-omics or streaming time series (markets,
sensors). Most of the computation can be avoided in most cases, while particular data is still
needed many times.

| id | feature | status | needs | priority | source | why |
|---|---|---|---|---|---|---|
| O1 | Lazy factor nodes: a stub with a store key that loads its factor and parameters on first use (a message, a free energy), with an LRU cache and eviction; the graph's structure stays resident | planned | O2, O3 | medium | Daniel, 2026-11-05 | a model whose parameters do not fit in memory; mixture-of-experts serving does this for its experts (offloading, memory-mapped weights) |
| O2 | Nodes stored as description plus arrays: a registered type tag and version, the constructor's arguments as data, and `ps`/`st` in a memory-mappable format (Arrow, SafeTensors, HDF5/JLD2 datasets, `Mmap`); no serialised functions or closures (Julia's `Serialization` is not a stable format, and deserialising untrusted data is unsafe) | planned | — | medium | Daniel, 2026-11-05 | factors are already split the Lux way, a small description and arrays; only `init_*` closures hold code, and they are needed only for initial parameters |
| O3 | A store interface with back ends: files keyed by node id, LMDB (memory-mapped, read-heavy), SQLite, RocksDB, Redis, object stores | planned | O2 | low | Daniel, 2026-11-05 | one interface; the choice per deployment |
| O4 | The subgraph a query reaches: from the clamps and the query variables, the factors connected through unobserved variables (Bayes-ball / d-separation), and a schedule restricted to it; incremental appends (iSAM-like) touch only the newest factors | planned | — | medium | Daniel, 2026-11-05; LenticulumLM's Incremental Next-Token Prediction | what makes lazy loading pay; full loopy inference would load everything |
| O5 | Parameters by key on demand instead of one graph-wide `NamedTuple` (the integer-slot store of the inference section), and training on a loaded subgraph with write-back of parameters and optimiser state | planned | I5, O1 | low | Daniel, 2026-11-05; ChatGPT review | gradients are local per parameter key, so a loaded subgraph trains as a block of coordinates |
