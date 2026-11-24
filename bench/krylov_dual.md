# Krylov solves with a ForwardDiff right-hand side (P42)

`krylov_dual.jl` compares solvers on dense real systems. With
$W = \operatorname{randn}(n,n)/(4\sqrt n)$, the GMRES pair uses $A = I + W$,
and both KrylovJL Hermitian methods and the in-house MINRES use $A = I + W^\top W$.
Each fixture has one primal right-hand side and $N$ seeded random partials.
The script checks the value and every tangent against a dense solve before
timing. Both tolerances are $10^{-10}$. Time is the minimum of 20 warm calls.
Bytes are the allocations of one warm call, and `nfe` counts operator products
for the full solve, including any tangent corrections.

Run with `julia -t auto --project=bench bench/krylov_dual.jl`. The smoke runner
uses size 20, one or two partials and one timed call per solver.

The measurement used Julia 1.13.1 and Krylov.jl 0.10.12 on an AMD Ryzen 7 7800X3D
with eight Julia threads and four BLAS threads. The uncommitted P42 working tree
was based on `02b3092cdf34cb5ad42637142dace8dca0b334ff`.

Each cell gives warm time in milliseconds, allocated bytes and `nfe`, in that order.

| n | N | GMRES | KrylovJL GMRES | MINRES | KrylovJL MINRES | KrylovJL CG |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 100 | 1 | 0.092 / 190,384 / 22 | 0.051 / 153,632 / 37 | 0.037 / 93,344 / 10 | 0.028 / 53,744 / 21 | 0.025 / 49,200 / 21 |
| 100 | 4 | 0.328 / 447,880 / 30 | 0.140 / 446,896 / 92 | 0.149 / 216,624 / 10 | 0.085 / 173,048 / 51 | 0.075 / 161,688 / 51 |
| 100 | 8 | 1.152 / 819,616 / 43 | 0.285 / 861,616 / 166 | 0.524 / 378,672 / 10 | 0.175 / 354,584 / 91 | 0.158 / 334,136 / 91 |
| 1,000 | 1 | 4.000 / 1,301,088 / 19 | 1.696 / 1,206,184 / 39 | 1.995 / 884,704 / 10 | 0.995 / 447,192 / 21 | 0.968 / 414,072 / 21 |
| 1,000 | 4 | 19.275 / 2,808,624 / 19 | 4.819 / 3,583,416 / 96 | 9.835 / 2,085,472 / 10 | 2.992 / 1,467,992 / 51 | 2.964 / 1,385,192 / 51 |
| 1,000 | 8 | 100.780 / 4,818,584 / 19 | 11.888 / 6,977,336 / 172 | 52.138 / 3,686,624 / 10 | 8.660 / 3,053,336 / 91 | 8.593 / 2,904,296 / 91 |

At size 1,000 with eight partials, KrylovJL GMRES took 11.888 ms and
172 operator products. In-house GMRES took 100.780 ms and 19 products.
The separate solves keep the Krylov iterations in Float64. The KrylovJL
residual for the tangent correction is the only dual matrix-vector product
in these fixtures. In-house GMRES allocates fewer bytes at size 1,000 with
four or eight partials. The table measures dense CPU solves and allocations
per call, including workspace construction.
