#implementation

> One root-finder serves both directions of a DEQ, because both directions are the *same
> residual* solved for a different variable. That is the entire reason a DEQ is worth
> expressing as a factor.

> Sources: code: `solve.jl`
>
> Theory (CT-ML wiki): [Bayesian Inversion](https://mathstruct.org/CategoryTheory-ML-Wiki/Bayesian-Inversion) · [Bayesian Lens](https://mathstruct.org/CategoryTheory-ML-Wiki/Bayesian-Lens) · [Lens](https://mathstruct.org/CategoryTheory-ML-Wiki/Lens)

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

Both root algorithms are derivative-free, so inference has no hard AD dependency.
Optional AD of a solved state is a separate IFT calculation. These are also what
`DeepEquilibriumNetworks.jl` does — Broyden and limited-memory Broyden are its workhorses.

## 2. `SolveReport`, and why nothing throws

A non-convergent solve returns `converged = false` and the caller decides. That is
[[Inversions and Bayesian Lenses]]'s position quoted directly: *a solver that stopped early is simply an
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

> [!important] A singular $I - \partial_z g$ prevents applying the IFT
> It means the IFT does not establish a locally differentiable inverse; the relation may branch
> there, but uniqueness alone is not ruled out. The algebraic family calls that locus the discriminant
> ([[Branches and the Discriminant]]); it is literally the same phenomenon in a different
> model class, and it produces the same failure. Two of the three
> [[Implicit Learners]] families therefore fail in the same place for the same reason, which
> the vault had not previously connected.

## 4. Implementation difficulties

### 4.1 Optional AD uses the implicit function theorem

`fd_jacobian` remains a forward-difference reference. `deq_sensitivity(...; wrt=:ps,
ad)` and the differentiable `solve_state(...; ad)` use the chosen DI backend instead.
For a residual $r(z,u)=0$ they solve

$$
D_u z^* = -(D_z r)^{-1} D_u r.
$$

The ChainRulesCore weak extension supplies an `rrule` for the internal solved-state
primitive. A cotangent $v$ gives $\lambda$ by $(D_z r)^\top\lambda=v$, then
$\bar u=-(D_u r)^\top\lambda$ via a backend VJP. The ForwardDiff weak extension
strips duals before the root solve, obtains residual JVPs in their seed directions,
and solves the same linear system to rebuild the root's duals. Neither differentiates
Broyden/Picard iterations. Both the input and parameters are active; the initial guess,
solve report and returned Lux state are nondifferentiated bookkeeping.

AD requires a converged root and a nonsingular residual Jacobian. A singular Jacobian
means the IFT gives no local inverse; it does **not** prove nonuniqueness (for example
$r(z)=z^3$ has a unique root and zero derivative there). The solve is dense: AD products
remove parameter-Jacobian storage in the reverse rule, but the state Jacobian and its
factorisation still cost quadratic storage and cubic factorisation work. A matrix-free
Krylov version remains future work. These are first-order rules; higher derivatives
through the primitive, the reverse `solve_input`, GPU and additional backends are untested.

`test/parameter_ad.jl` checks ForwardDiff and Zygote against the linear-cell closed
form and central differences, including a nonzero downstream scalar loss.

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

### 4.4 Root state is fixed during `solve_state`

A fixed-point solve must evaluate one fixed residual, including its incoming Lux state.
`solve_state` now holds that state fixed through the root iterations and advances it once
at the final root. This also defines the equation used by the IFT. State is not an
additional implicit variable. A cell that mutates arrays in its incoming state in place
violates this functional contract. The reverse `solve_input` still has the legacy `Ref`
threading and has no AD rule.

Related: [[deq]], [[flow]], [[neuralode]], [[The Equilibrium Family]],
[[Backpropagation by the Implicit Function Theorem]], [[Branches and the Discriminant]]

Solved-state IFT AD currently accepts `AutoForwardDiff()` and `AutoZygote()`. Other
solve backends are rejected explicitly, so they cannot silently differentiate the
root iterations. Pass the backend to `solve_state` as well as the outer loss
derivative. Core pointwise parameter derivatives have wider backend support.
