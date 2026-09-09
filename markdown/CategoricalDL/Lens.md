# Lens — bidirectional information flow

> Cruttwell et al., Definition 2.4.

## The problem it solves

Learning is bidirectional. Predictions flow forward; corrections flow backward. A lens is
the minimal categorical gadget that packages *one arrow's worth* of both directions
together, so that composing lenses automatically composes both passes.

## Definition

For a Cartesian category $\mathcal{C}$, the category $\mathbf{Lens}(\mathcal{C})$ has

- **objects**: pairs $(A, A')$ of objects of $\mathcal{C}$. Think: $A$ = values,
  $A'$ = changes in values. They need not be equal (these are *bimorphic* lenses).
- **maps** $(A,A') \to (B,B')$: pairs $(f, f^*)$ with
  - $f : A \to B$ — the **get** (forward, prediction)
  - $f^* : A \times B' \to A'$ — the **put** (backward, correction)

  Note the shape of the `put`: it needs the *original input* $A$ as well as the incoming
  change $B'$. This is exactly why backpropagation must cache activations.
- **identity** on $(A,A')$: $(1_A, \pi_1)$ — pass the change straight through.
- **composition** of $(f,f^*)$ and $(g,g^*)$: get is $f\,;\,g$, put is
  $$\langle \pi_0,\; \langle \pi_0\,;\,f,\; \pi_1\rangle\,;\,g^* \rangle\,;\,f^*$$

## Reading the composition formula

That put formula is opaque in point-free notation. Pointwise, with $a \in A$ and
$c' \in C'$:

$$(g \circ f)^*(a, c') \;=\; f^*\bigl(a,\; g^*(f(a),\, c')\bigr)$$

That is: run the forward pass to get $b = f(a)$, push the correction $c'$ back through
$g^*$ *at the point $b$*, then push the result back through $f^*$ *at the point $a$*. This
is precisely the reverse-mode chain rule. The formula is not an analogy for
backpropagation — for the right choice of $\mathcal{C}$ it *is* backpropagation, and the
choice is [[Cartesian Reverse Differential Category]].

## Example — a `Dense` layer's lens

Take $f(x) = Wx$, so $(A,A') = (\mathbb{R}^n, \mathbb{R}^n)$ and
$(B,B') = (\mathbb{R}^m, \mathbb{R}^m)$:

$$f(x) = Wx, \qquad f^*(x, \bar{y}) = W^\top \bar{y}$$

and composing two of them gives $f_1^*(x, W_2^\top \bar z) = W_1^\top W_2^\top \bar z$,
which is the chain rule. The $x$ argument is unused here because the map is linear; put
$\tanh$ in and it is used.

## The structure is monoidal but not Cartesian

$(A,A') \otimes (B,B') := (A \times B,\; A' \times B')$. But $\mathbf{Lens}(\mathcal{C})$
is *not* Cartesian: there is no terminal object, because a lens into $(T,T)$ would need a
unique `put` $A \times T \to A'$ and there are many. **You cannot delete a backward wire.**
This is the formal reason `Lens` cannot silently drop gradients, and the reason a
computation graph over lenses stays balanced.

## Graphical calculus

A lens is drawn as a box with a wire in each direction on each side:

```
        A ──►┌──────────┐──► B
             │ (f, f*)  │
       A' ◄──└──────────┘◄── B'
```

Composition is joining the $B/B'$ wires. This picture is the single most useful mental
image in the whole theory — every object in Lenticulum is one of these boxes.

Related: [[Para]], [[Parametric Lens]], [[Bayesian Lens]]
