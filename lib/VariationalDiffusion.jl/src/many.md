#implementation

> Batched implicit inference shares field and Jacobian evaluations across queries.
> Each query keeps its own convergence and step decisions.
>
> Sources: [[implicit]] §7, [[Shared Solver Layer]] D3. Code in `many.jl`. The test
> “many queries at once: the same answers, shared network calls” in `test/runtests.jl`
> compares the batch against individual solves.
>
> Theory (CT-ML wiki): —

`implicit_infer_many` assigns each query a column and groups the active columns by noise
level for network evaluations. It uses local Newton and trust-region implementations.
Each output is an `ImplicitSolution` with a state vector in the working element type and
a residual in its primal type, following the same reporting convention as single queries.
