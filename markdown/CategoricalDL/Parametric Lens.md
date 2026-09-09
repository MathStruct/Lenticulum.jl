# Parametric Lens — Definition 2.5

> Cruttwell et al., Definition 2.5. **This is the object a Lux.jl layer is.**

## Definition

$\mathbf{Para}(\mathbf{Lens}(\mathcal{C}))$, the category of **parametric lenses**:

- **objects**: pairs $(A, A')$ of objects of $\mathcal{C}$;
- **morphisms** $(A,A') \to (B,B')$: a choice of parameter *pair* $(P, P')$ together with a
  lens
  $$(f, f^*) : (A,A') \otimes (P,P') \longrightarrow (B,B')$$
  which unpacks to two ordinary maps
  $$f : P \times A \to B \qquad\qquad f^* : P \times A \times B' \to P' \times A'$$

Everything about supervised learning is in that second signature. Feed it a parameter, an
input, and a desired change in the output; get back the change in the parameter *and* the
change in the input.

## The picture

$$
\begin{array}{c}
\quad P \;\downarrow\quad \uparrow\; P' \\[2pt]
A \;\rightarrow\; \boxed{\;f,\ f^*\;} \;\rightarrow\; B \\[2pt]
A' \;\leftarrow\; \phantom{\boxed{\;f,\ f^*\;}} \;\leftarrow\; B'
\end{array}
$$

Three wires in each direction: data left-to-right, corrections right-to-left, parameters
top-down and updates bottom-up.

## Composition

```tikz
\usepackage{tikz}
\begin{document}
\begin{tikzpicture}[font=\small]
  % --- box f ---
  \draw[rounded corners] (0,-0.8) rectangle (2.0,0.8);
  \node at (1.0,0) {$f,\ f^*$};
  \draw[->] (-1.6,0.35) -- (0,0.35);   \node at (-1.9,0.35) {$A$};
  \draw[<-] (-1.6,-0.35) -- (0,-0.35); \node at (-1.9,-0.35) {$A'$};
  \draw[->] (2.0,0.35) -- (3.6,0.35);
  \draw[<-] (2.0,-0.35) -- (3.6,-0.35);
  \draw[->] (0.7,2.0) -- (0.7,0.8);    \node at (0.45,1.6) {$P$};
  \draw[<-] (1.3,2.0) -- (1.3,0.8);    \node at (1.55,1.6) {$P'$};
  % --- box g ---
  \draw[rounded corners] (3.6,-0.8) rectangle (5.6,0.8);
  \node at (4.6,0) {$g,\ g^*$};
  \node at (2.8,0.35) {$B$};  \node at (2.8,-0.35) {$B'$};
  \draw[->] (5.6,0.35) -- (7.2,0.35);   \node at (7.5,0.35) {$C$};
  \draw[<-] (5.6,-0.35) -- (7.2,-0.35); \node at (7.5,-0.35) {$C'$};
  \draw[->] (4.3,2.0) -- (4.3,0.8);     \node at (4.05,1.6) {$Q$};
  \draw[<-] (4.9,2.0) -- (4.9,0.8);     \node at (5.15,1.6) {$Q'$};
\end{tikzpicture}
\end{document}
```

Join the $B/B'$ wires; the parameter wires of both boxes stay dangling upward and become
the composite's parameter wire $(Q \times P,\; Q' \times P')$.

## Why the parameter pair $(P,P')$ and not just $P$

Because the update to a parameter need not live in the same space as the parameter. For
$\mathbf{Smooth}$ they coincide ($P' = P = \mathbb{R}^p$), and this is the case everyone
has in mind. But for Boolean circuits over $\mathbb{Z}_2$ the "gradient" is an XOR mask;
for a parameter constrained to a manifold (a rotation, a covariance matrix, a simplex) the
update lives in the tangent space, not in the manifold. Keeping $P'$ distinct is what lets
the same framework cover the natural-gradient / Fisher-metric case that
[[Parameterized Statistical Game]] needs.

Lenticulum takes this seriously: the energy space $E$ of [[Scalar and Multivariate Energy]]
is a genuinely separate object from the value spaces, for exactly this reason.

## What is *still missing* for learning

A parametric lens takes a change in $B$ and returns a change in $P$. But when you train,
you have a *target value* $b \in B$, not a change $b' \in B'$; and you want a *new
parameter*, not a change in parameter. Bridging those two gaps is what the loss map, the
learning rate and the optimiser are for — see
[[Learning Components as Parametric Lenses]].

Related: [[Para]], [[Lens]], [[Lux as a Parametric Lens]], [[Statistical Game]]
