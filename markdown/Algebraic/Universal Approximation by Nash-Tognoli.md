# Universal Approximation by Nash–Tognoli

> [[README]]'s table claims the implicit analogue of Weierstraß is "compact smooth manifolds
> via Nash–Tognoli". That is right — but the analogy is weaker than it looks in three
> specific ways, and each one is a live research gap.

## The theorem

**Nash (1952) / Tognoli (1973).** Every **compact smooth manifold without boundary** is
diffeomorphic to a **nonsingular real algebraic variety**.

Nash proved that $M$ is diffeomorphic to a union of connected components of a real algebraic
set; Tognoli removed the "union of components" caveat, giving the clean statement. There is
also an approximation form: a compact smooth submanifold can be $C^\infty$-approximated by
nonsingular algebraic subvarieties, subject to the caveat in §3 below.

So the expressive-power claim is genuine: the model class of
[[The Veronese Parametrisation|varieties $V(\Theta v_d)$]] is rich enough to represent, up to
diffeomorphism, any compact smooth manifold — which is exactly the class of relations one
would want an implicit learner to reach.

## Gap 1: no degree bounds — the approximation is not *effective*

Weierstraß comes with quantitative companions: Jackson's theorem bounds the degree needed to
achieve error $\varepsilon$ in terms of the modulus of continuity, and modern neural-network
approximation theory has an entire literature of rate theorems.

**Nash–Tognoli has nothing of the kind.** It is an existence statement. There is no known
bound, in terms of any geometric invariant of $M$ (dimension, curvature, reach, volume,
topological complexity), on the degree $d$ required for an algebraic model — let alone one
for approximating $M$ to within $\varepsilon$.

Since $d$ controls $m = \binom{N+d}{d}$ and therefore *everything* (parameter count, sample
complexity, Bézout count, elimination cost — see
[[The Veronese Parametrisation]] §"The number that kills it"), **the absence of degree bounds
means there is no theory at all of when this model class is practical for a given target.**

> [!warning] This is the largest theoretical gap in the family
> "Universal approximation" is doing much less work here than the phrase suggests. Compare:
> for neural networks we can say "width $O(\varepsilon^{-n})$ suffices"; here we can say
> only "some degree suffices". Establishing an effective Nash–Tognoli — a degree bound in
> terms of the reach or the topology of $M$ — would be a genuine research contribution and I
> am not aware of one. See [[Open Problems in Algebraic Implicit Learning]] §3.

## Gap 2: the algebraic model may not live in *your* ambient space

Nash–Tognoli gives a variety **diffeomorphic** to $M$, in *some* $\mathbb{R}^{N'}$. It does
not say that a smooth submanifold $M \subseteq \mathbb{R}^N$ can be approximated by an
algebraic subvariety **of that same $\mathbb{R}^N$**, isotopically.

That stronger statement is genuinely obstructed. The obstruction is homological: a
$\mathbb{Z}/2$ homology class of a real algebraic set that is represented by an algebraic
subset must lie in the subgroup of **algebraic cycles**, and this subgroup can be proper
(Borel–Haefliger). There are smooth submanifolds whose homology class is not algebraic, and
they therefore cannot be isotoped to an algebraic subvariety of the ambient space.

For us this matters because **the data live in a fixed $\mathbb{R}^N$**, and the whole point
of the implicit formulation is to learn the relation *in the data's own coordinates*. If the
relation you want is not algebraically representable there, the theorem does not help — you
would need to lift to a higher-dimensional space, i.e. introduce latent channels, which
changes the model.

This is a real and under-appreciated caveat, and it is the algebraic-geometry counterpart of
"you may need a wider network".

## Gap 3: nonsingular is what the theorem gives; singular is what fitting produces

Nash–Tognoli produces a **nonsingular** variety. Nothing in
[[Fitting is a Nullspace Problem]] produces one. The fitted $V(\Theta^\star v_d)$ is
generically singular somewhere, and its singular locus is where
[[Branches and the Discriminant|everything fails at once]].

There is no known way to constrain the eigenproblem to nonsingular varieties — smoothness is
a semialgebraic condition on $\Theta$ (the Jacobian has full rank at every real point of the
variety) and is not expressible as a linear constraint. A moment–SOS certificate of
smoothness would be a principled route but has a moment matrix of the usual prohibitive size.

## Gap 4: the real locus can be much smaller than the theorem's object

Even granting all of the above, a variety fitted to data can have
$\dim_\mathbb{R} V \ll \dim_\mathbb{C} V$, or $V_\mathbb{R}$ can be nearly a finite set — see
[[Varieties Ideals and Real Nullstellensatz]] §"Problem 1". Nash–Tognoli concerns
$V_\mathbb{R}$ as a manifold; the fitting procedure controls only the complex ideal. The two
are connected by nothing.

## What survives

Despite four gaps, the qualitative claim in [[README]] is correct and worth keeping:

| | explicit | implicit |
|---|---|---|
| approximated object | continuous functions on a compact set | compact smooth manifolds |
| theorem | Weierstraß / Stone–Weierstraß | Nash–Tognoli |
| effective? | **yes** (Jackson rates) | **no** |
| in the ambient space? | yes | **not necessarily** (Borel–Haefliger) |

The right reading is: *the model class is expressive enough in principle; we have no theory
of how expensive that expressiveness is.* Which, given [[The Veronese Parametrisation]]'s
combinatorics, is precisely the question that decides whether the family is usable.

Related: [[Varieties Ideals and Real Nullstellensatz]], [[The Veronese Parametrisation]], [[Open Problems in Algebraic Implicit Learning]], [[Implicit Learners]]
