# Para — parameters, categorically

> Cruttwell et al., Definition 2.1–2.3.

## The problem it solves

A neural network layer is *not* a function $A \to B$. A `Dense(3 => 5)` is a function

$$\mathbb{R}^{20} \times \mathbb{R}^3 \longrightarrow \mathbb{R}^5$$

taking 20 numbers of weights and 3 numbers of input. But we still want to *think* of it as
an arrow $\mathbb{R}^3 \to \mathbb{R}^5$, because that is how we wire layers together — we
do not thread the parameter wire through the composition by hand. `Para` is the bookkeeping
that lets you say "arrow $A \to B$" while a parameter object rides along.

## Definition

For a strict symmetric monoidal category $\mathcal{C}$ with product $\otimes$ and unit $I$,
the category $\mathbf{Para}(\mathcal{C})$ has

- **objects**: the objects of $\mathcal{C}$;
- **maps** $A \to B$: pairs $(P, f)$ with $P$ an object of $\mathcal{C}$ and
  $f : P \otimes A \to B$;
- **identity** on $A$: $(I, 1_A)$;
- **composition** of $(P,f) : A \to B$ with $(P', f') : B \to C$:
  $$(P' \otimes P,\; (1_{P'} \otimes f)\,;\,f')$$

Read the composition rule carefully — it is the whole point. **Parameter spaces multiply.**
Composing a 20-parameter layer with a 30-parameter layer gives one 50-parameter arrow, and
the composite's parameter wire is the *pair* of the two parameter wires.

## Example

$\mathcal{C} = \mathbf{Smooth}$: objects are natural numbers, a map $n \to m$ is a smooth
function $\mathbb{R}^n \to \mathbb{R}^m$. Then

$$\texttt{Dense(3 => 5, tanh)} \;=\; (\mathbb{R}^{20},\; (W,b,x) \mapsto \tanh(Wx+b))
\;:\; 3 \longrightarrow 5$$

is a morphism of $\mathbf{Para}(\mathbf{Smooth})$, and

$$\texttt{Chain(Dense(3 => 5), Dense(5 => 2))} = (\mathbb{R}^{12} \otimes \mathbb{R}^{20}, \ldots) : 3 \to 2$$

Note $\mathbb{R}^{12} \otimes \mathbb{R}^{20}$, in that order: the *second* layer's
parameters come first in the composite, exactly as written in the composition rule. Lux.jl
stores this as `(layer_1 = ..., layer_2 = ...)` — a `NamedTuple` is a product with labels,
which is a strictly better data structure than $P' \otimes P$ because it is
order-insensitive. See [[Lux as a Parametric Lens]].

## Reparametrisation (Definition 2.3)

A **reparametrisation** of $(P, f) : A \to B$ by a map $\alpha : Q \to P$ is

$$(Q,\; (\alpha \otimes 1_A)\,;\,f) : A \to B$$

i.e. you feed the parameter wire through $\alpha$ first. This is a 2-cell of
$\mathbf{Para}(\mathcal{C})$, which makes $\mathbf{Para}$ a *bicategory*, not a category.

Reparametrisation looks like a technicality; it is not. **Optimisers are
reparametrisations** — see [[Learning Components as Parametric Lenses]] §4. Weight tying,
hypernetworks, LoRA and quantisation are all reparametrisations too.

```tikz
\usepackage{tikz-cd}
\begin{document}
\begin{tikzcd}[row sep=large, column sep=huge]
Q \otimes A \arrow[r, "\alpha \otimes 1_A"] \arrow[dr, "(Q,\,\alpha;f)"'] & P \otimes A \arrow[d, "f"] \\
& B
\end{tikzcd}
\end{document}
```

## Where this shows up in Lenticulum

`Para` is the reason a Lenticulum factor carries `ps` (parameters) separately from its
inputs, and the reason `initialparameters` returns a nested `NamedTuple` mirroring the
factor tree. It is inherited unchanged from LuxCore.

Related: [[Lens]], [[Parametric Lens]], [[Parameterized Statistical Game]]
