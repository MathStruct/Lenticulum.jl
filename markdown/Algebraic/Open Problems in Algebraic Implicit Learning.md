# Open Problems in Algebraic Implicit Learning

> The honest ledger. What in this family is solved, what is a matter of engineering, and
> what is genuinely open. Ordered by how much they block progress.

## 1. Controlling the *real* locus — **open, and blocking**

Everything computable in the pipeline — the residual, the eigenproblem
([[Fitting is a Nullspace Problem]]), the Jacobian rank, the Bézout count — is a statement
about the **complex** variety. Everything you want — that the relation is a nonempty
manifold of the intended dimension, in the data's own coordinates — is a statement about
$V_\mathbb{R}$. By [[Varieties Ideals and Real Nullstellensatz]] the two can be arbitrarily
far apart: $x^2+y^2$ defines a complex curve whose real locus is a point.

**Nothing in the fitting procedure prevents this.** The fit sees only the data, and is free
to return an ideal whose real locus is barely larger than the training set — which is a very
precise form of overfitting, invisible to the training objective.

*What would fix it:* a tractable regulariser that penalises real-codimension exceeding the
complex codimension. Positivstellensatz / moment–SOS certificates are the only principled
route known, and their moment matrices carry the same $\binom{N+\rho}{\rho}$ cost. **I am not
aware of any practical method.** This is the deepest gap.

## 2. The $\binom{N+d}{d}$ wall — **open; several attacks, none sufficient**

$m = \binom{N+d}{d}$ is simultaneously the parameter count, the sample complexity, the
eigenproblem size, and the base of every downstream blowup
([[The Veronese Parametrisation]]). Attacks that have been tried:

| approach | why it helps | why it is not enough |
|---|---|---|
| **sparse supports** — restrict to a chosen monomial set | $m$ becomes the support size; BKK bounds tighten | choosing the support is the model-selection problem in disguise; getting it wrong is unrecoverable |
| **kernelisation** — the polynomial kernel computes $\langle v_d(x), v_d(x')\rangle$ in $O(N)$ | fitting depends on data only through $S = V^\top V$, which is Gram-like | the *output* is $\Theta$, a vector in the $m$-dimensional feature space; you cannot root-find in an implicit feature space, so inference is lost |
| **random projection / sketching** of the Veronese space | reduces $m$ with JL guarantees on the fit | destroys the algebraic structure: a projected ideal is not an ideal, so elimination, resultants and root counts are all unavailable |
| **low-rank / tensor structure** on $\Theta$ | fewer parameters, respects symmetry | restricts the model class in a way with no approximation theory behind it |
| **toric / hierarchical restriction** | genuinely better behaved everywhere ([[Algebraic Statistics Bridge]]) | only applies when the structure is known a priori |

The kernelisation row is the instructive one: **the fitting problem kernelises and the
inference problem does not.** That asymmetry looks fundamental — inference needs explicit
coefficients to run a solver — and it is the single most valuable thing to attack. If someone
found a way to do root finding in an implicit feature space, the family would scale.

## 3. Effective Nash–Tognoli — **open**

[[Universal Approximation by Nash-Tognoli]]: no bound is known on the degree $d$ needed to
represent or approximate a given compact manifold, in terms of any geometric invariant
(reach, curvature, topological complexity, volume). Contrast Jackson's theorem for
Weierstraß, or the width bounds for neural approximation.

Since $d$ controls $m$ controls everything, **there is currently no theory of when this model
class is affordable for a given target.** Even a crude bound — "$d = O(\text{something about
the reach})$" — would convert the family from folklore to engineering.

Also open in the same direction: the Borel–Haefliger obstruction means a relation may not be
algebraically representable *in the data's ambient space* at all. There is no diagnostic for
this, and no theory of how many latent channels would suffice to lift out of it.

## 4. Model selection is a jump between manifolds — **open**

Choosing $(k, d)$ means choosing a Grassmannian $\mathrm{Gr}(k, \binom{N+d}{d})$. Different
choices are different manifolds of different dimension with no smooth path between them
([[The Parameter is a Grassmannian]] §"Caveat"). The numerical-rank decision of
[[Fitting is a Nullspace Problem]] is therefore a **discrete jump**, controlled by a
tolerance $\tau$ with no principled setting, and the output is discontinuous in $\tau$.

*What would fix it:* a continuous relaxation — a nuclear-norm or log-det surrogate on the
Veronese moment matrix $S$ whose solution path in a regularisation parameter is continuous
and whose knots recover the discrete choices. Plausible, and I am not aware of it having been
done for the vanishing-ideal problem specifically.

## 5. Approximate vanishing ideals are ill-posed — **partly solved**

Gröbner bases are discontinuous in their coefficients, so they cannot be applied to fitted
floating-point $\Theta$ ([[Composition is Elimination]] §"Failure 4"). **Border bases**
(Kehrein–Kreuzer, Mourrain) and the AVI family of algorithms
(Heldt–Kreuzer–Pokutta–Poulisse; VCA; GPCA) solve the *stability* problem.

What remains: the tolerance parameter (§4), and the absence of any **statistical** theory —
there is no consistency result of the form "with $M$ samples from a distribution supported on
$V$ and noise $\sigma$, the recovered ideal converges to $I(V)$ at rate $\ldots$". Given how
much is known about the corresponding question for PCA and subspace recovery, this looks
attackable rather than deep.

## 6. Composition is not closed — **understood, and correctly designed around**

The composite of two algebraic relations is only **constructible** (Chevalley) /
**semialgebraic** (Tarski–Seidenberg), so the category is not closed under composition
without taking Zariski closures, which lose information
([[Composition is Elimination]], [[Algebraic Statistics Bridge]]).

This is not open — it is a theorem — and the architectural response is correct and already
in place: **do not compose, schedule messages**. The quantitative payoff ($L\,d^q$ versus
$d^{Lq}$) is the sharpest justification for `Mycelium.jl` anywhere in the vault. Worth
recording as a *solved* problem so it is not re-litigated.

## 7. The discriminant — **intrinsic, not fixable**

Branch collisions make inference discontinuous, gradients unbounded, and the entropy a step
function ([[Branches and the Discriminant]]). This is a property of *relations*, not of the
algebraic representation: $y = \pm\sqrt{1-x^2}$ has the same behaviour however you write it.
Any implicit learner that permits multi-valued relations inherits it.

The correct engineering response is **detection and reporting** — $\kappa(J_u)$ is an exact,
cheap proximity indicator — rather than a fix. What *is* open is the right *loss* behaviour
near $\Delta$: damping the adjoint solve is standard practice with no theory.

Related open sub-question: **monodromy**. Branches cannot in general be labelled consistently
along a cycle in the factor graph, so message passing around a loop may return to a different
branch than it started on. I am not aware of any treatment of this in the implicit-layer
literature, and it is a genuine correctness issue for cyclic graphs — arguably the most
concrete new problem this note set surfaces.

## 8. Geometric-distance fitting — **understood, expensive**

Exact ML fitting needs the geometric distance, whose evaluation is an EDD-many root find *per
data point* ([[Algebraic versus Geometric Distance]]). Sampson + IRLS is the practical
answer; it works well empirically and has no convergence proof. Not a research emergency, but
the absence of a proof should be stated rather than glossed.

## Summary table

| # | problem | status |
|---|---|---|
| 1 | controlling the real locus | **open, blocking** |
| 2 | the $\binom{N+d}{d}$ wall (esp. kernelising *inference*) | **open, blocking** |
| 3 | effective Nash–Tognoli degree bounds | **open** |
| 4 | continuous model selection over $(k,d)$ | **open, probably tractable** |
| 5 | statistical consistency of approximate vanishing ideals | **open, probably tractable** |
| 5b | numerical stability of AVI | solved (border bases) |
| 6 | non-closure under composition | theorem; designed around |
| 7 | discriminant discontinuity | intrinsic; detect, do not fix |
| 7b | monodromy on graph cycles | **open, and specific to this project** |
| 8 | geometric-distance fitting | expensive; IRLS unproven |

## The one-line recommendation

Build this family as the **reference implementation and test oracle** — the case where every
abstract slot of [[The Algebraic Factor as a Statistical Game|the statistical game]] is
computable exactly, so the framework itself can be validated — and not as the production
path. For production, its role is niche and genuine: low-dimensional factors with known
polynomial structure (kinematics, multi-view geometry, reaction networks) embedded in a graph
whose other factors are neural.

Related: [[Algebraic Implicit Learners]], [[The Algebraic Factor as a Statistical Game]], [[Implicit Learners]]
