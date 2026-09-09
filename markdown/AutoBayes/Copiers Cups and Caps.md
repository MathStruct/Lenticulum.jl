# Copiers, Cups and Caps — Remark 8

> AutoBayes, §2 (unnumbered), Remark 8. **This is where the DAG restriction dies.**

## Copier

As a kernel $A \rightsquigarrow A \times A$:

$$\text{\Large$\wedge$}(da_1, da_2 \mid a_0) \;=\; [a_1 = a_0 = a_2]\, da_1\, da_2$$

It enforces that two consumers of a variable see the *same* value. In a Bayesian network
drawing this is the invisible fact that an arrow can fan out; in a string diagram it is an
explicit node, which is better, because in a general Markov category copying is *not* free
and making it explicit is what keeps you honest.

## Cup and cap

$$\mathrm{cup}_A(da_1, da_2) = [a_1 = a_2]\,da_1\,da_2 \;:\; 1 \nrightarrow\!\!\!\bullet\; A \otimes A$$
$$\mathrm{cap}_A(a_1, a_2) = [a_1 = a_2] \;:\; A \otimes A \nrightarrow\!\!\!\bullet\; 1$$

(The cap is an unnormalised kernel $A \times A \rightsquigarrow 1$, equivalently a function
$A \times A \to [0,\infty)$.)

**The cup turns an unobserved space into an observed one; the cap does the reverse.**

Remark 8: this makes the bicategory of open models **self-dual compact closed**.

## Why you care: this is how you bend wires

In a DAG framework, "input" and "output" are fixed by the graph. Compact closure says they
are not: given $c : X \nrightarrow\!\!\!\bullet\; Y$ you can slide the $X$ leg around the cup to get
$1 \nrightarrow\!\!\!\bullet\; X \otimes Y$, in which $X$ is now *observed*. Nothing about $c$ changed —
only which of its legs you are holding.

This is precisely the "no distinguished input/output direction" claim in [[README]], and
precisely the $P_{in}/P_{out}/P_{latent}$ selection of [[ImplicitREDDiff]]. See
[[Channels and Polarity]] for the reconciliation.

```tikz
\usepackage{tikz}
\begin{document}
\begin{tikzpicture}[font=\small]
  % left: c : X -> Y, X unobserved
  \draw[rounded corners] (0,-0.5) rectangle (1.2,0.5);
  \node at (0.6,0) {$c$};
  \draw[<-] (0,0) -- (-1.4,0); \node at (-1.7,0) {$X$};
  \draw[->] (1.2,0) -- (2.6,0); \node at (2.9,0) {$Y$};
  \node at (0.6,-1.6) {unobserved $X$, observed $Y$};
  % right: X bent around a cup, now observed
  \draw[rounded corners] (7.0,-0.5) rectangle (8.2,0.5);
  \node at (7.6,0) {$c$};
  \draw[->] (8.2,0) -- (9.6,0); \node at (9.9,0) {$Y$};
  \draw (7.0,0) -- (6.2,0);
  \draw (6.2,0) arc (90:270:0.5);
  \draw[->] (6.2,-1.0) -- (9.6,-1.0); \node at (9.9,-1.0) {$X$};
  \node at (7.9,-2.0) {both observed, via $\mathrm{cup}_X$};
\end{tikzpicture}
\end{document}
```

## Cyclic models

Allowing unnormalised measures, the acyclicity constraint is relaxed. The paper's example
is the joint

$$[a_1 = a_2]\; r(da_2, dc \mid b)\; q(db \mid a_1)\; da_1$$

which as a "generalized Bayesian network" would be an ambiguous diagram with $A \to B \to A$,
but as a string diagram with a cup is unambiguous. **Feedback loops, DEQs, and mutually
recursive latent variables all live here.**

> [!warning] The price
> Unnormalised measures mean the "distribution" flowing on a wire may not have mass 1, and
> the normalising constant is exactly the thing that is expensive to compute. Cycles buy
> you expressiveness and hand you back a partition-function problem. This is the same trade
> as an energy-based model versus a normalising flow, and it is why the *energy* (which is
> defined pointwise, no normalisation needed) is the object Lenticulum optimises — see
> [[Scalar and Multivariate Energy]].

## Where this is used

The paper's Example 4 (supervised learning): compose $X \otimes c : X \otimes X \multimap X \otimes Y$
after $\mathrm{cup} : 1 \multimap X \otimes X$, and now *both* $X$ and $Y$ are observed. That is
supervised learning: you know the labels. The cup is the categorical name for "clamp this
variable to data".

Related: [[Open Model]], [[Channels and Polarity]], [[Examples from the Paper]]
