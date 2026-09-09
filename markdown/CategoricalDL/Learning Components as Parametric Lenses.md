# Learning Components as Parametric Lenses

> Cruttwell et al., §3. Model, loss, learning rate, optimiser — all one shape.

The payoff of §2 is that the four things you name when you set up a training run are all
morphisms in the *same* category, and "setting up training" is *composition*.

## 3.1 Model

A model is a $\mathbf{Para}(\mathcal{C})$-map $(P, f) : A \to B$ pushed through
$\mathbf{Para}(R)$ (see [[Cartesian Reverse Differential Category]]), giving the parametric
lens $(f, R[f])$. In Lux: a layer plus its `pullback`.

## 3.2 Loss map (Definition 3.3)

> A **loss map** on $B$ consists of a $\mathbf{Para}(\mathcal{C})$-map
> $(\mathrm{loss}, B) : B \to L$ for some object $L$.

Read that parameter space: **the loss map's parameter is the label.** This is not a trick.
It is the observation that a loss function $B \times B \to \mathbb{R}$, where one argument
is the prediction and the other is the ground truth, is naturally a $B$-parametrised map
$B \to \mathbb{R}$. Putting labels and weights on the same footing is what later lets the
loss itself be learned (GANs), and — in AutoBayes — is what lets *data* be just another
factor in a graph.

Four worked examples from the paper:

| loss | $L$ | $e(b_t, b_p)$ | $R[e](b_t, b_p, \alpha)$ |
|---|---|---|---|
| quadratic | $\mathbb{R}$ | $\tfrac12\sum_i ((b_p)_i - (b_t)_i)^2$ | $\alpha \cdot (b_p - b_t,\; b_t - b_p)$ |
| Boolean | $\mathbb{Z}_2^b$ | $b_t + b_p$ (XOR) | $(\alpha, \alpha)$ |
| softmax cross-entropy | $\mathbb{R}$ | $\sum_i (b_t)_i\bigl((b_p)_i - \log \mathrm{softmax}(b_p)_i\bigr)$ | — |
| dot product | $\mathbb{R}$ | $b_t \cdot b_p$ | $\alpha\cdot(b_p,\; b_t)$ |

Composing model with loss gives a lens $A \to L$ whose parameters are $(P, B)$ — weights
*and* label.

## 3.3 Learning rate (Definition 3.8)

> A **learning rate** $\alpha$ on $L$ is a lens $(L, L') \to (1,1)$.

The `get` must be the unique map to the terminal object, so all the content is in the
`put`, a map $\alpha^* : L \to L'$. It **caps off** the dangling $L/L'$ wire on the right
of the diagram — the loss goes out, and a scalar seed comes back in.

- Standard supervised learning in $\mathbf{Smooth}$: $\alpha^*(l) = -\epsilon$, a constant.
  (The minus sign is where "descent" enters — it is here, and only here.)
- Boolean circuits: $\alpha^*(l) = l$, the identity.
- Loss-dependent rate: $\alpha^*(l) = -\epsilon\, l$.

That the constant-$\epsilon$ learning rate is a *cap* — a map that discards the loss value
and injects a constant — is a small revelation: **the numerical loss value is never used by
gradient descent.** Only its derivative is. The loss number exists for you, not for the
algorithm.

## 3.4 Optimiser as reparametrisation (Definitions 3.11, 3.14)

After capping, the composite is a lens whose only dangling wires are $(P, P')$: it takes a
parameter and returns a parameter *update*. We want a parameter. So we need a box on top of
the $P/P'$ wires — a lens $(P,P) \to (P,P')$ — which is a [[Para|reparametrisation]].

**Gradient update** (Def. 3.11): $G(p) = p$, $G^*(p, p') = p + p'$.

**Stateful update** (Def. 3.14): a state object $S$ and a lens
$U : (S \times P,\; S \times P) \to (P, P')$.

| optimiser | $S$ | $U(s,p)$ | $U^*(s,p,p')$ |
|---|---|---|---|
| gradient descent | $1$ | $p$ | $p + p'$ |
| momentum | $P$ | $p$ | $(s',\; p + s')$, $s' = -\gamma s + p'$ |
| Nesterov | $P$ | $p + \gamma s$ | $(s',\; p + s')$, $s' = -\gamma s + p'$ |
| AdaGrad | $P$ | $p$ | $(g',\; p + \tfrac{\epsilon}{\delta + \sqrt{g'}} \odot p')$, $g' = g + p'\odot p'$ |

Nesterov is the example that justifies the machinery: it is the first optimiser whose lens
has a **non-trivial `get`** ($p + \gamma s$, not $p$). "Evaluate the gradient at the
look-ahead point" is literally "the forward part of the optimiser lens is not the
identity". No other formalism makes that as obvious.

## The complete supervised learning system

```tikz
\usepackage{tikz}
\begin{document}
\begin{tikzpicture}[font=\small]
  % model
  \draw[rounded corners] (0,-0.8) rectangle (1.9,0.8);
  \node at (0.95,0) {model};
  \draw[->] (-1.5,0.35) -- (0,0.35);   \node at (-1.8,0.35) {$A$};
  \draw[<-] (-1.5,-0.35) -- (0,-0.35); \node at (-1.8,-0.35) {$A'$};
  % loss
  \draw[rounded corners] (3.1,-0.8) rectangle (5.0,0.8);
  \node at (4.05,0) {loss};
  \draw[->] (1.9,0.35) -- (3.1,0.35);   \node at (2.5,0.65) {$B$};
  \draw[<-] (1.9,-0.35) -- (3.1,-0.35); \node at (2.5,-0.65) {$B'$};
  \draw[->] (3.75,2.0) -- (3.75,0.8);   \node at (3.5,1.6) {$B$};
  \draw[<-] (4.35,2.0) -- (4.35,0.8);   \node at (4.6,1.6) {$B'$};
  % learning rate cap
  \draw[rounded corners] (6.2,-0.8) rectangle (7.3,0.8);
  \node at (6.75,0) {$\alpha$};
  \draw[->] (5.0,0.35) -- (6.2,0.35);   \node at (5.6,0.65) {$L$};
  \draw[<-] (5.0,-0.35) -- (6.2,-0.35); \node at (5.6,-0.65) {$L'$};
  % optimiser above the model
  \draw[rounded corners] (0,2.0) rectangle (1.9,3.4);
  \node at (0.95,2.7) {optimiser};
  \draw[->] (0.65,2.0) -- (0.65,0.8);   \node at (0.35,1.4) {$P$};
  \draw[<-] (1.25,2.0) -- (1.25,0.8);   \node at (1.6,1.4) {$P'$};
  \draw[->] (0.65,4.4) -- (0.65,3.4);   \node at (0.1,4.0) {$S{\times}P$};
  \draw[<-] (1.25,4.4) -- (1.25,3.4);   \node at (1.9,4.0) {$S{\times}P$};
\end{tikzpicture}
\end{document}
```

Every wire that is still dangling is something *you* supply: the input $A$, the label $B$
(on the loss's parameter wire), and the optimiser state $S \times P$. Every wire that got
joined is something the framework handles. A training loop is: run the whole composite
lens's `get`, then its `put`, and you get back new optimiser state and new parameters.

Related: [[Parametric Lens]], [[Lux as a Parametric Lens]], [[Composition of Gradients]]
