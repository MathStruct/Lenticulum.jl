# Energy-Based Factor Graphs

> §6 of [[Energy-Based Learning]] is titled *Efficient Inference: Non-Probabilistic Factor
> Graphs*. It is `Mycelium.jl` without the beliefs.
>
> And it settles a complaint this vault has filed three times in three different packages.

## 1. The structure, and it is the same structure

LeCun's energy-based factor graph:

$$E(Y, X) \;=\; \sum_c E_c\bigl(X,\, Y_{S_c}\bigr),
\qquad
Y^\ast \;=\; \operatorname*{arg\,min}_Y \sum_c E_c$$

Energies **add** over factors; inference **minimises** the sum. On a tree the minimisation
factorises and the algorithm is **min-sum** — Viterbi — with exactly the message structure of
sum-product and a different semiring.

| LeCun §6 | Mycelium |
|---|---|
| a factor $E_c$ | a factor |
| a variable in $S_c$ | an edge |
| $E = \sum_c E_c$ | `GradedEnergy`'s $\oplus$; the Bethe sum |
| min-sum messages | `factor_message` with `DiracBelief`s |
| min-out a latent | `Latent()` polarity |
| no partition function | `combine` returns unnormalised beliefs |
| tree ⇒ exact | `istree(g)` ⇒ `tree_schedule` is exact |

The last row holds in both semirings and for the same reason: on a tree the exclusion
principle makes each message summarise a disjoint subtree, and that argument never mentions
which operations are being used.

## 2. The reframing: a Dirac message is a min-sum message

Every factor in every `lib/` package returns a `DiracBelief`.

| package | factor | returns |
|---|---|---|
| `VariationalDiffusion` | `DiffusionFactor` | `DiracBelief` (RED-Diff's $q$ is a point mass) |
| `ImplicitLayers` | `DEQFactor` | `DiracBelief` (a root-find returns a point) |
| `ImplicitLayers` | `NeuralODEFactor` | `DiracBelief` (an ODE solve returns a point) |
| `Adversarial` | `GeneratorFactor` | `SampleBelief` — particles, still no density |

Three notes have recorded this as a defect, each blaming the same cause — `GaussianBelief`
being stranded in the top-level package ([[The Equilibrium Family]] §5). That diagnosis is
still true as far as it goes. But it is not the whole story, and the framing was wrong:

> [!important] Those factors are not broken Bayesian factors. They are correct energy-based
> factors.
> A `DiracBelief` message is a min-sum message. A root-find, an ODE solve and a RED-Diff prox
> are all **minimisations**, and a minimisation is what an energy-based factor graph does for
> inference. Three independent packages converged on point-valued messages not because each
> hit the same missing type, but because **each wraps a model whose native inference is
> $\arg\min$**.
>
> The project has an energy-based layer and a Bayesian layer, and most of the implementation
> lives in the first one.

## 3. Min-sum is the zero-temperature limit

The two semirings are not unrelated. Put a temperature on the Gibbs distribution,
$p(y) \propto e^{-E(y)/T}$, and note

$$-T\log\sum_i e^{-E_i/T} \;\xrightarrow[T\to 0]{}\; \min_i E_i$$

so **sum-product at temperature $T$ becomes min-sum as $T\to 0$**. Marginalisation becomes
minimisation; the Gibbs distribution concentrates on the argmin; a belief becomes a Dirac.

Two consequences worth writing down.

### 3.1 Latents may be minimised out, and at $T=0$ that is exact

[[Energy-Based Learning]] §5 gives both readings of a latent variable, and the min version is
the $T\to0$ limit of the marginal version.

`constraint.md` §4.1 records latent-channel marginalisation as "the single most valuable
missing piece" in that file, and `Copiers Cups and Caps` calls marginalisation "the expensive
one". In a Dirac-valued graph the cheap alternative is not an approximation at all — it is
the correct operation for the semiring the graph is actually running in.

### 3.2 The Bethe free energy degenerates *correctly*

Three packages have recorded a version of: *these factors contribute energy but no entropy, so
the counting correction has nothing to correct and the total is not $-\log p(y)$.*

Under the temperature reading that is not a bug report. The Bethe free energy is

$$F \;=\; \underbrace{\sum_c F_c}_{\text{energies}} \;+\; \underbrace{\sum_v (1-d_v)\,H_v}_{\text{entropies}}$$

and the entropy term carries a factor of $T$. At $T = 0$ it vanishes — regardless of $H_v$
being $-\infty$ for a point mass, because $T\cdot H \to 0$ — and what is left is

$$F \;=\; \sum_c E_c$$

which is **LeCun's energy, exactly**. So a graph of Dirac-valued factors is not failing to
compute a free energy; it is computing the right object at the wrong temperature for the
surrounding framework.

> [!warning] The real problem is the temperature mismatch, not the missing entropy
> Lenticulum's Bethe form has no explicit $T$ — it is implicitly $T = 1$. The Dirac factors
> are operating at $T = 0$ inside it. Mixing them with a `GaussianFactor` sums a
> zero-temperature energy and a unit-temperature free energy and calls the result a free
> energy.
>
> That is a sharper statement of the defect than "these factors have no entropy", and it
> suggests a different fix: **a temperature per factor, or a declared semiring per graph**,
> rather than forcing every factor to produce a distribution it does not have.

## 4. What this does and does not resolve

**Resolved:** the three complaints about Dirac-valued inversions were describing one thing —
an energy-based sublayer — and describing it as a deficiency. It is a legitimate mode of
operation with its own theory, its own exactness result on trees, and its own literature.

**Not resolved:** whether the project *wants* $T = 0$. The case for $T > 0$ is everything
AutoBayes is for — posteriors, uncertainty, the free energy as a model-comparison score,
[[The Linear Gaussian Chain]] §4's identity $F = -\log p(y)$ that holds to machine precision.
None of that survives at zero temperature.

So `GaussianBelief` belonging in `LenticulumCore` is still the right call
([[The Equilibrium Family]] §5). What changes is the urgency and the reason: not "three
packages are broken" but "three packages are energy-based and there is currently no way to be
anything else from a `lib/` package."

**Newly visible:** the framework has no way to *say* which semiring a graph is running in,
which [[The Type Discipline of a Factor Graph]] §3.3 identifies as a **missing type index** — if
a belief carried its semiring, mixing would not typecheck, and the choice would be forced at
construction rather than remembered.
`istree`, `isdag` and `isexact` are all present; a `MinSum` / `SumProduct` distinction is not.
A graph mixing the two is silently wrong, and nothing in `validate(g)` looks.

## 5. Loopy min-sum is a different subject

Off a tree the two semirings diverge, and the guarantees are not the same ones.

[[Loopy Message Passing]] and [[The Linear Gaussian Chain]] discuss loopy *sum-product*: for
Gaussians, exact means and wrong variances (Weiss & Freeman), which
`ImplicitLayers`'s resistive-divider test exhibits with a stable factor of five.

Loopy **min-sum** has its own analysis, and the flavour of the result is different: a fixed
point of max-product/min-sum is locally optimal in a neighbourhood generated by trees and
single loops, which is a genuine guarantee but a weaker and differently-shaped one than
"the means are exact". A graph that switches semirings switches which theory applies to it,
and the vault currently documents only one of them.

## 6. What would follow

Not a plan — a note of what the reading suggests, in rough order of how much it buys:

1. **Declare the semiring.** A trait on the graph or the factor saying whether messages are
   min-sum or sum-product, checked in `validate`. Cheap, and it turns a silent error into a
   loud one.
2. **A temperature.** $F = E - T\,H$ with $T$ explicit makes §3.2's degeneration a limit
   rather than an inconsistency, and makes annealing expressible — which is what
   `ImplicitREDDiff`'s $\rho$ weights already are in disguise.
3. **`min` as an alternative to marginalisation for `Latent()`.** Cheap, correct at $T = 0$,
   and it unblocks `constraint.md` §4.1 in the regime those factors actually run in.
4. **Loss functionals.** The largest and the one [[Energy-Based Learning]] §3 argues is
   missing outright.

Related: [[Energy-Based Learning]], [[Training Energy-Based Models]], [[Factor Graphs]],
[[The Type Discipline of a Factor Graph]],
[[Bethe Free Energy]], [[Messages are Inversions]], [[Schedules]],
[[Loopy Message Passing]], [[The Equilibrium Family]], [[The Linear Gaussian Chain]]
