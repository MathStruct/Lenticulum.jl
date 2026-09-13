# The Table Revisited

> [[README]] opens with a nine-row table contrasting explicit and implicit learning. It was
> written before any of the vault existed, it is quoted by seventeen notes, and it is kept
> verbatim for that reason. This note reads it row by row against what the vault has since
> found — because several rows are now known to be loose, and one is wrong in a way that
> matters.
>
> The verdict up front: the table describes the **algebraic family** and the **LeCun layer**
> only — polynomials, varieties, a squared residual. It has no row for beliefs, and it says
> nothing about the AutoBayes layer that is the actual architecture. That is not a mistake in
> any row; it is a mistake in what the table is a table *of*.

## Row by row

### Approximator: *multivariate polynomials* vs *algebraic varieties*

**Scoped, not wrong.** This is the algebraic family of [[Algebraic Implicit Learners]] and
nothing else. A `DEQFactor` is not a variety; a `DiffusionFactor` is a score, not an ideal;
a `LinearConstraintFactor` is a variety but a degree-one one. The row is true of one of
[[Three Senses of Implicit]]'s three senses and silent on the other two.

The honest general row would be: *functions* vs *relations*, with "variety" as the special
case where the relation is polynomial.

### Inference: *forward evaluation* vs *rootfinding*

**True**, and [[Inference as Root Finding]] is the note. One refinement the vault added:
rootfinding is the $T \to 0$, Dirac-belief case. With Gaussian beliefs inference is
marginalisation, and the root is the mean of a distribution rather than a point
([[Energy-Based Factor Graphs]] §3). The row is the min-sum reading of a sum-product
architecture.

### Backpropagation: *reverse-mode AD* vs *implicit function theorem / differential algebra*

**Half right.** The IFT does the backward pass, exactly —
[[Backpropagation by the Implicit Function Theorem]], and `deq_sensitivity` in
`ImplicitLayers` is that theorem as code. Differential algebra does a *different* job:
index reduction, hidden constraints, identifiability
([[Differential Algebra and DAE Factors]]). Pairing the two with a slash reads as though
they were alternatives for the same task, and they are not.

There is one narrow place where the pairing is vindicated: for a DAE of index $\nu \ge 2$
the adjoint is itself index-$\nu$, index reduction must be run on both forward and adjoint
consistently, and that reduction *is* differential algebra inside the backward pass. Narrow,
but real — and it is the only sense in which the row is right.

### Universal approximation: *Weierstraß* vs *Nash–Tognoli*

**A loose analogy between two different kinds of theorem.** Weierstraß is an
*approximation* statement: for every $\varepsilon$ there is a polynomial within $\varepsilon$,
of degree growing without bound. Nash–Tognoli is an *exact* statement: every compact smooth
manifold is *diffeomorphic* to a nonsingular real algebraic variety — not approximated by
one. [[Universal Approximation by Nash-Tognoli]] spells this out. They rhyme; they are not
the same shape of result.

What the analogy hides is that **degree is the quantity that matters**, and the table has no
row for it. [[Depth in Implicit Learning]] shows the degree of a composed chain is $d^L$,
which is simultaneously the cost of flattening and the expressivity gained by not flattening.
The table's silence on degree is the table's silence on depth.

### Well-posedness: *always single-valued* vs *multi-valued, no solution, closest point*

**The best row in the table.** It is the observation El Ghaoui's implicit deep learning
does not make and this framework does — that a relation need not have a solution, may have
several, and that the honest output is then a projection onto the variety rather than a
point on it ([[Algebraic versus Geometric Distance]], [[Branches and the Discriminant]]).
Everything in the well-posedness literature of the repository grows from this row.

### Loss formulation: $\|f_\theta(x)-y\|^2$ vs $\|r_\theta(x_1,\dots,x_n)\|^2$

**Substantively wrong, and this is the one that matters.** $\|r_\theta\|^2$ alone is what
LeCun calls the *energy loss*: it pushes the energy down on the data and pushes nothing up
anywhere else. If $r_\theta$ has any freedom to become small everywhere — and a learned
residual does — the minimiser is the trivial relation $r \equiv 0$, which relates everything
to everything and says nothing. [[Energy-Based Learning]] §3–4 has this as the collapse
problem, and [[Training Energy-Based Models]] catalogues the three standard second terms
(MCMC, score matching, NCE) that prevent it.

The explicit column does not have this problem, because $y$ is observed and $f_\theta$
cannot make $\|f(x)-y\|^2$ small except by matching it. The implicit column inherits the
form and loses the guarantee. So the row is not "the same loss, relational" — it is a loss
that works in one column and collapses in the other, and the table does not say so.

The AutoBayes answer is the **free energy**, not the residual norm: $\|r\|^2$ enters as one
term, and the normaliser (or its variational bound) supplies the contrastive one
([[Statistical Game]], [[Energy-Based Factor Graphs]]). The row should read
$-\log p_\theta(\text{data})$ or its bound, and $\|r\|^2$ is the Gaussian, fixed-precision,
single-factor special case.

### Symmetry handling: *fixed unidirectional output* vs *symmetric: no distinguished input/output*

**Wrong word, and a claim stronger than the code.** "Symmetric" in mathematics means
$R(x,y) \Leftrightarrow R(y,x)$ — the relation is its own converse — and almost no
relation of interest is that. $y = x^2$ is not. What the row means is **undirected** or
**acausal**: no channel is designated the input. [[Channels and Polarity]] is the right
vocabulary, and "polarity is a property of the query, not the factor" is the right slogan.

And the claim is stronger than most of the implementation delivers. A `GaussianFactor`
supports both polarities of a two-channel relation — it is *bidirectional*. Only
`LinearConstraintFactor` is genuinely acausal in the $n$-channel sense, with every
polarity of every channel available ([[The Structural Gap to ModelingToolkit]]). "No
distinguished input/output" is the design goal; "either channel may be the output" is what
most factors actually do.

Nothing in the row is about symmetry in the geometric-deep-learning sense — a group acting
on the data — and [[Geometric Deep Learning and Physical Laws]] §7 records that this project
has none of that.

### Computational cost: *cheap* vs *expensive (Newton, etc.)*

**True and unquantified.** [[Composition is Elimination]] has the numbers: elimination is
doubly exponential in the worst case, message passing on a tree is linear in the number of
factors, and the $130\times$ and $O(n^2)$ findings of [[Parallelism and Compilation]] are
the implementation's own costs, which have nothing to do with Newton. The row's "etc." is
carrying most of the content.

### Layer connections: *DAG* vs *arbitrary connected graph*

**True.** With the caveat that "arbitrary" is where every difficulty of the vault lives:
the moment the graph has a cycle, message passing is no longer elimination
([[Loopy Message Passing]], [[The Linear Gaussian Chain]] §5), and the free energy
needs the Bethe correction. A DAG is not merely a restriction; it is the case where
inference is exact for free.

## What is not in the table

| missing row | explicit | implicit | where the vault has it |
|---|---|---|---|
| **Uncertainty** | none — a point prediction | a belief over every channel | [[messages]], [[Channels and Polarity]] |
| **Composition** | function composition, always defined | relational composition, closed only for $d=1$ | [[Composition is Elimination]] |
| **Depth** | layers | degree $d^L$, or solver iterations, or eliminated latents | [[Depth in Implicit Learning]] |
| **Training objective** | the loss | the free energy, with a contrastive term | [[Energy-Based Learning]] |
| **Time** | none | none — and that is the gap to MTK | [[Time as a Base]] |

The first row is the one that makes the table a table of the wrong thing. The architecture
is AutoBayes: factors are statistical games, channels carry beliefs, inference is message
passing, learning is free-energy minimisation. Every row of the original table is a
statement about the $T \to 0$, Dirac-belief, single-factor limit of that — the LeCun layer,
where energies are residual norms and inference is rootfinding. That limit is the right way
to *explain* the idea, and it is what the table does well. It is not the thing that was
built.

## Summary

Three rows are fine (Inference, Well-posedness, Layer connections). Two are true but say
less than they seem (Approximator, Cost). Two are loose (Backpropagation, Universal
approximation). Two are wrong in the letter — one harmlessly (Symmetry: wrong word), one
not (Loss: collapses without a second term). And the table as a whole omits beliefs, which
is the difference between what it describes and what exists.

The table stays in the README as written, because the notes that cite it cite the idea,
and the idea was right.

Related: [[README]], [[Implicit Learners]], [[Three Senses of Implicit]],
[[Energy-Based Learning]], [[Channels and Polarity]], [[Depth in Implicit Learning]],
[[Differential Algebra and DAE Factors]], [[Universal Approximation by Nash-Tognoli]],
[[Geometric Deep Learning and Physical Laws]]
