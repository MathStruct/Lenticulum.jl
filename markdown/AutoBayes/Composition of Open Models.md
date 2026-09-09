# Composition of Open Models — Definitions 4–7

> AutoBayes, Definitions 4 and 6, Remarks 5 and 7.

## Sequential (Definition 4)

Given $p : X \nrightarrow\!\!\!\bullet\; Y$ and $q : Y \nrightarrow\!\!\!\bullet\; Z$, the composite
$q \circ\!\!\!\bullet\; p : X \nrightarrow\!\!\!\bullet\; Z$ has latent space

$$\llbracket q \circ\!\!\!\bullet\; p \rrbracket \;=\; \llbracket p \rrbracket \times Y \times \llbracket q \rrbracket$$

and kernel

$$(q \circ\!\!\!\bullet\; p)(ds, dy, dt, dz \mid x) \;=\; q(dt, dz \mid y)\; p(ds, dy \mid x)$$

**No integral sign.** Compare the usual kernel composite
$(q \bullet p)(dz|x) = \int_y q(dz|y)\,p(dy|x)$ — that one has an integral, and is why
Bayesian networks are expensive to compose. Open models pay a memory cost instead.

Note also: if $p$ and $q$ are both pure, $q \circ\!\!\!\bullet\; p$ is *not* pure. Purity is not
preserved by composition, which is exactly why the latent space had to be introduced.

**Identity** (Remark 5): $\mathrm{id}_X(dx \mid x') = [x = x']\,dx$, the Dirac kernel; pure.

This yields a **bicategory** whose 1-cells are open models — bicategory, not category,
because associativity holds only up to the isomorphism reassociating the latent products.

## Parallel / tensor (Definition 6)

For $q : Y \nrightarrow\!\!\!\bullet\; Z$ and $q' : Y' \nrightarrow\!\!\!\bullet\; Z'$:

$$\llbracket q \otimes q' \rrbracket = \llbracket q \rrbracket \times \llbracket q' \rrbracket$$
$$(q \otimes q')(dt, dt', dz, dz' \mid y, y') = q(dt, dz \mid y)\, q'(dt', dz' \mid y')$$

Monoidal product on the bicategory, unit $1$ (Remark 7).

## The two derived operations you will actually use

### `reveal`

The latent space is just a factor of the kernel's codomain, so it can be *promoted* to the
observed space by a purely formal move (a 2-cell):

$$\mathrm{reveal}_{A}\bigl(q \circ\!\!\!\bullet\; p\bigr) \;:\; 1 \nrightarrow\!\!\!\bullet\; A \otimes B$$

This costs nothing. It is a retyping, not a computation. In Lenticulum this is how you
expose an intermediate activation as an observable channel for probing, debugging, or
attaching an auxiliary loss.

### Dummy variables

$\mathrm{id}_A \otimes q : A \otimes X \nrightarrow\!\!\!\bullet\; A \otimes Y$, abbreviated $A \otimes q$.
This lets information **flow past a factor** untouched. It is the categorical name for a
skip connection / residual bypass, and for the "this factor does not depend on that
variable" pattern in a factor graph.

## Every Bayesian network is a composite of open models

Following Fong (2013, Thm 4.5): topologically sort the nodes so $j < i$ whenever there is
no directed path $i \to j$. For each node take
$p_i : \bigotimes_{j \in \mathrm{pa}(i)} X_j \nrightarrow\!\!\!\bullet\; X_i$, reveal the parents (so
later factors can see them), pad with dummy variables for the non-parents, i.e. form

$$\Bigl(\bigotimes_{j<i,\, j \notin \mathrm{pa}(i)} X_j\Bigr) \otimes \mathrm{reveal}_{\mathrm{pa}(i)}(p_i)
\;:\; \bigotimes_{j<i} X_j \nrightarrow\!\!\!\bullet\; \bigotimes_{j \le i} X_j$$

then compose in order. So the expressiveness is at least that of Bayesian networks — and,
via [[Copiers Cups and Caps]], strictly more.

> [!warning] This is the algorithm Mycelium.jl has to implement
> "Topologically sort, reveal parents, pad with dummies, compose in sequence" is a concrete
> compilation procedure from a factor graph to a string of composable open models. For
> *cyclic* graphs it does not apply and you need cups/caps plus a message-passing schedule
> instead. That split — acyclic ⇒ compile to a sequence, cyclic ⇒ schedule messages — is
> the central design fork in Mycelium.

Related: [[Open Model]], [[Copiers Cups and Caps]], [[Composition of Bayesian Lenses]]
