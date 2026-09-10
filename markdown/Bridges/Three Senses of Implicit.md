# Three Senses of Implicit

> [[Implicit Learners]] defines *implicit* one way. Two well-known papers use the word for two
> **other** things, and neither is wrong — they are different axes.
>
> This note separates them, because conflating them is how you end up expecting a GAN to run
> backwards.

## 1. The three senses

| | what is implicit | you cannot | the paper |
|---|---|---|---|
| **(1) implicit likelihood** | the **density** $p_\theta(x)$ | evaluate $p_\theta(x)$ | Mohamed & Lakshminarayanan, [1610.03483](https://arxiv.org/pdf/1610.03483) |
| **(2) implicit computation** | the **value** of the output | write $y$ as a closed-form expression | El Ghaoui et al., [1908.06315](https://arxiv.org/abs/1908.06315) |
| **(3) implicit relation** | the **direction** | say which channel is the input | [[README]], [[Implicit Learners]] |

Sense (1): you can sample $x = G_\theta(z)$ but there is no likelihood. The model is still a
*function* $z \mapsto x$.

Sense (2): $y$ is defined by a fixed-point equation, $x = \phi(Ax + Bu)$, $y = Cx + Du$. You
solve rather than evaluate. The model is still a *function* $u \mapsto y$.

Sense (3): there is no input. $r_\theta(x_1,\ldots,x_n) \approx 0$, and which channels you
solve for is a call-site decision — [[Channels and Polarity]].

## 2. They are independent axes

| model | (1) likelihood | (2) computation | (3) direction |
|---|---|---|---|
| `Dense` / a Lux `Chain` | — | — | — |
| **GAN generator** | **✓** | — | — |
| normalising flow | — | — | — |
| **DEQ**, `NeuralODEFactor` | — | **✓** | — |
| VAE decoder | — | — | — |
| **`LinearConstraintFactor`** | — | — | **✓** |
| `GaussianFactor` | — | — | ✓ (two directions) |
| an algebraic variety factor | — | ✓ | ✓ |

Three observations from that table:

- **A normalising flow is implicit in no sense at all** — tractable density, closed-form
  evaluation, fixed direction — despite being a generative model. Generativeness is not
  implicitness.
- **A DEQ scores (2) and not (3)**, which is precisely what
  `ImplicitLayers.jl` is about: `DeepEquilibriumNetwork` gives you (2), and keeping the
  residual instead gives you (3) as well. [[DEQ as a Relation]].
- **A GAN generator scores (1) and nothing else.** It is a function, evaluated in closed form,
  in one direction. Which is why `Adversarial.GeneratorFactor` has exactly one polarity and
  says so.

## 3. On *Implicit Deep Learning* (1908.06315)

El Ghaoui, Gu, Travacca, Askari and Tsai define the **implicit prediction rule**

$$x = \phi(Ax + Bu), \qquad y = Cx + Du$$

which generalises a feedforward net's recursion into a single fixed-point equation over one
hidden vector. It is a good paper and the notational simplification is real.

It is sense (2), and the reason it stops there is not an oversight — **it is the paper's
central technical contribution.**

> [!important] Well-posedness is exactly the assumption that kills sense (3)
> Much of the paper is about *well-posedness*: conditions (a Perron–Frobenius bound on $|A|$,
> for instance) guaranteeing that the fixed-point equation has a **unique** solution $x$ for
> every input $u$ — and those conditions are then **imposed during training**, so the learned
> rule is guaranteed to satisfy them.
>
> A relation with a unique solution for every input **is a function.** So the paper's
> well-posedness programme is precisely the work of ensuring that the implicit relation
> collapses back to an explicit map.

Compare [[README]]'s own table, which goes the other way on the same row:

| | explicit | implicit |
|---|---|---|
| **well-posedness** | always single-valued | *may be multi-valued or have no solution / output only closest point to the variety* |

So the two projects take opposite positions on the same question. El Ghaoui et al. *engineer
away* multi-valuedness because it obstructs a prediction rule. Lenticulum *keeps* it, because
a relation that branches is still a relation — and the branch locus has a name and a theory
([[Branches and the Discriminant]]), and energy minimisation is total precisely where
root-finding is not ([[Implicit Learners]] §"The claim").

That is the full scope the title does not reach: not a criticism of the paper, but a statement
of where its assumptions place it on the table in §1.

## 4. On *Learning in Implicit Generative Models* (1610.03483)

Mohamed & Lakshminarayanan's sense is (1), and it is the sharpest of the three because it is
about what you can *compute* rather than about how the model is written:

> A model is implicit if it defines a **sampling procedure** and not a density.

Their consequence is the interesting part, and it is worked through in
[[Implicit Generative Models]]: without a likelihood you cannot do maximum likelihood, so you
must learn **by comparison** — estimate a density ratio or difference between $p^\ast$ and
$q_\theta$ and drive it to one. GANs are one of four ways to do that.

Note what this shares with sense (3) and what it does not. Both give up a closed-form
objective. But an implicit *generative* model gives up the **density** while keeping the
**direction**; an implicit *learner* gives up the **direction** while (often) keeping a
perfectly computable residual. `LinearConstraintFactor` has an entirely tractable density and
no preferred direction; a GAN generator is the exact opposite.

## 5. Why this matters for the vault

[[Implicit Learners]] lists three model *families* — algebraic, equilibrium, diffusion — and
each is implicit in sense (3). The adversarial family added by `lib/Adversarial.jl` is **not**:
it is sense (1), and all three of its factors are unidirectional.

So the honest structure is two-dimensional, and the vault had been treating it as one:

- **sense (3)** is what makes a factor worth having a `Polarity` at all, and what
  `Mycelium.jl` exists to schedule;
- **sense (1)** is what makes a belief a `SampleBelief` rather than a parametric one, and it
  is what breaks `Mycelium.combine`.

A model can be either, both, or neither, and the two failures land in different parts of the
codebase.

Related: [[Implicit Learners]], [[Implicit Generative Models]], [[GANs as Two Factors]],
[[DEQ as a Relation]], [[The Equilibrium Family]], [[Channels and Polarity]],
[[Branches and the Discriminant]], [[The Structural Gap to ModelingToolkit]]
