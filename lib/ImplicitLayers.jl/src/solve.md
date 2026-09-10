# solve.jl — implementation note

> One root-finder serves both directions of a DEQ, because both directions are the *same
> residual* solved for a different variable. That is the entire reason a DEQ is worth
> expressing as a factor.

## 1. Two solvers, and the difference is the caveat

$$
\text{Picard: } u \leftarrow u - \beta F(u)
\qquad\qquad
\text{Broyden: } u \leftarrow u - H F(u),\quad H \approx J^{-1}
$$

For the DEQ residual $F(z) = z - g(z,x)$, Picard *is* "keep applying the layer":
$z \leftarrow (1-\beta)z + \beta g(z,x)$. It converges **iff the iteration is a
contraction**, which is [[Implicit Learners]]'s recorded caveat —
*"this only works if the iteration converges, and unconstrained DEQs need not"* — and the
test suite makes it concrete: a linear cell with $\rho(W) = 1.6$ has a perfectly good unique
fixed point that Picard cannot reach and Broyden reaches in a handful of steps.

Broyden is initialised with $H_0 = I$, which makes its first step

$$u_1 = u_0 - I\cdot F(u_0) = u_0 - (u_0 - g(u_0)) = g(u_0)$$

**exactly a Picard step**, with everything after it the correction Picard never makes.
Asserted in the tests, and worth knowing because it means Broyden is never worse than Picard
on the first iteration.

Both are derivative-free, which is a requirement rather than a preference: the residual
contains a neural network and this package has no AD dependency. It is also what
`DeepEquilibriumNetworks.jl` does — Broyden and limited-memory Broyden are its workhorses.

## 2. `SolveReport`, and why nothing throws

A non-convergent solve returns `converged = false` and the caller decides. That is
[[Bayesian Lens]]'s position quoted directly: *a solver that stopped early is simply an
inexact inversion, and the loss records the cost.* Throwing would make a legal-but-poor
inversion into an error.

`nfe` is reported alongside `iterations` because iteration count is meaningless when a
Broyden step and a Picard step cost differently — `DeepEquilibriumNetworks.jl` reports it in
its `DeepEquilibriumSolution` for the same reason.

## 3. The implicit function theorem

$$
(I - \partial_z g)\frac{\partial z^\ast}{\partial x} = \partial_x g
\qquad\Longrightarrow\qquad
\frac{\partial z^\ast}{\partial x} = (I - \partial_z g)^{-1}\partial_x g
$$

**One linear solve, no unrolling** — the property that makes implicit layers $O(1)$ in memory
and the reason [[Implicit Learners]] lists the IFT as this family's backward pass. See
[[Backpropagation by the Implicit Function Theorem]].

`ift_sensitivity` throws `SingularException` when $I - \partial_z g$ is singular, and that
deserves emphasis:

> [!important] A singular $I - \partial_z g$ is not a numerical accident
> It is the statement that the fixed point is **not locally unique** — the relation branches
> there. The algebraic family calls that locus the discriminant
> ([[Branches and the Discriminant]]); it is literally the same phenomenon in a different
> model class, and it produces the same failure. Two of the three
> [[Implicit Learners]] families therefore fail in the same place for the same reason, which
> the vault had not previously connected.

## 4. Implementation difficulties

### 4.1 `fd_jacobian` is a test-scale tool, and says so

Dense forward differences: $n+1$ evaluations of $F$, and roughly half the significant digits
lost to the step size. For a real DEQ the answer is a **vector–Jacobian product from
automatic differentiation** at $O(1)$ cost.

It exists so the IFT can be *implemented and checked* — for a linear cell the exact
sensitivity is $(I-W)^{-1}U$ and the test suite compares against it. It is not a scalable
path, and `deq_sensitivity` inherits the limitation.

> The contrast with `VariationalDiffusion.jl` is the sharpest thing in this package.
> RED-Diff's stop-gradient meant that family needed **no derivative of the network at all**.
> The equilibrium family cannot do that: the IFT *is* a linear system built from the
> Jacobian, so the derivative is not an optimisation to be skipped, it is the answer. The
> diffusion family avoided AD by approximating; this one can only avoid it by not scaling.

### 4.2 Broyden stores a dense inverse Jacobian

$O(n^2)$ memory, which is fine at test scale and hopeless for a real DEQ. That is exactly
why `DeepEquilibriumNetworks.jl` ships *limited-memory* Broyden: keep the last $m$ secant
pairs instead of the matrix. Not implemented; it is a self-contained addition and the obvious
next one.

Anderson acceleration is likewise absent, and is the other solver DEQ implementations
commonly use.

### 4.3 The tolerance is absolute

`tol` compares against $\|F(u)\|_2$ with no relative component and no scaling by $\|u\|$. On
a badly scaled problem that is either unreachable or trivially satisfied. A real solver takes
`abstol` and `reltol`; `NonlinearSolve.jl` does, and if this package ever grows a dependency
it should be that one.

### 4.4 The state `Ref` leaks LuxCore's threading into the solver

`solve_root` wants a pure `F(u)`. LuxCore wants `st` threaded through every call. The factors
bridge that with a `Ref` captured in the closure, so the last call's state wins. For a
stateless cell that is correct; for a cell with running statistics (a `BatchNorm` inside a
DEQ) it is *whatever the final solver iterate happened to write*, which is not obviously the
right thing and is not obviously wrong either. Nothing tests it.

Related: [[deq]], [[flow]], [[neuralode]], [[The Equilibrium Family]],
[[Backpropagation by the Implicit Function Theorem]], [[Branches and the Discriminant]]
