#definition #category-theory #optics #computable

A **prism** is an [[optic]] for accessing optional or variant data, characterized as an optic for the [[cocartesian monoidal category|cocartesian monoidal structure]].

## Concrete Definition

For [[Object|objects]] $S$ (source), $T$ (target), $A$ (focus), $B$ (updated focus), a prism consists of:
- **Match**: $S \to A + R$ - attempts to extract the focus, returning either the focus or a remainder
- **Build**: $B \to T$ - constructs the target from the updated focus

where $+$ denotes the [[coproduct]].

## Coend Representation

A prism is formally defined as:
$$\text{Prism}(S, T, A, B) = \int^C \mathbf{C}(S, C + A) \times \mathbf{C}(C + B, T)$$

where the [[coend]] is taken over all objects $C$ in the [[cocartesian monoidal category]].

## Profunctor Representation

Prisms correspond to Tambara modules for the cocartesian tensor product:
$$\forall P. \text{Cocartesian}(P) \Rightarrow P(A, B) \to P(S, T)$$

## Prism Laws

- **Match-Build**: If $\text{match}(s) = \text{inl}(a)$, then $\text{build}(\text{match}(s)) = s$
- **Build-Match**: $\text{match}(\text{build}(b)) = \text{inl}(b)$ for all $b: B$

<!--
\begin{tikzcd}
S \arrow[r, "\text{match}"] & A + R \\
B \arrow[r, "\text{build}"] & T
\end{tikzcd}
-->

Prisms fall out when the monoidal structure is cocartesian, dual to how [[lens|lenses]] arise from [[cartesian monoidal category|cartesian monoidal structure]].