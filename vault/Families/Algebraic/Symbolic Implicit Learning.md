#open-problem

> Learn a relation $r(z) = 0$ as a **formula**: not a polynomial with fixed monomials, not a
> network, but a short symbolic expression found by search. It would make learned relations
> readable and comparable, and it is slow and hard. Noted for later; nothing is implemented.

> Sources: Schmidt & Lipson, *Distilling free-form natural laws from experimental data*, Science 2009; Mangan, Brunton, Proctor & Kutz, *Inferring biological networks by sparse identification of nonlinear dynamics*, IEEE TMBMC 2016 (implicit SINDy); Kaheman, Kutz & Brunton, *SINDy-PI*, Proc. R. Soc. A 2020; Cranmer, *Interpretable Machine Learning for Science with PySR and SymbolicRegression.jl*, [arXiv:2305.01582](https://arxiv.org/abs/2305.01582), 2023; Cranmer et al., *Discovering Symbolic Models from Deep Learning with Inductive Biases*, NeurIPS 2020
>
> Bibliography: [[Bibliography#^schmidt2009distilling|Schmidt & Lipson 2009]] · [[Bibliography#^mangan2016inferring|Mangan et al. 2016]] · [[Bibliography#^kaheman2020sindypi|Kaheman et al. 2020]] · [[Bibliography#^cranmer2023pysr|Cranmer 2023]] · [[Bibliography#^cranmer2020symbolic|Cranmer et al. 2020]]
>
> Theory (CT-ML wiki): [Statistical Game](https://mathstruct.org/CategoryTheory-ML-Wiki/Statistical-Game)

## 1. The idea

Symbolic regression searches over expression trees for $y = f(x)$. The implicit version
searches for $r$ with $r(z) = 0$ on the data, so that the formula is a **relation** with no
privileged output, the way this family's polynomials are ([[Algebraic Implicit Learners]]).
The algebraic family is the special case of a fixed library of monomials, where fitting is a
nullspace problem ([[Fitting is a Nullspace Problem]]); implicit SINDy is the same with any
fixed library of candidate terms. Symbolic implicit learning drops the fixed library.

## 2. Why it would be worth it

- **Readable relations.** The output is a law, $x^2 + y^2 - 1 = 0$, not a weight vector.
- **Model selection** is native: a Pareto front of complexity against residual is a family of
  candidate relations, and the free energy is a principled way to choose among them
  ([[Open Problems in Algebraic Implicit Learning]] §4 is the same jump-between-models problem).
- **Distillation.** A learned factor (a pair potential, an energy network) can be explained
  afterwards by a formula, as in Cranmer et al. 2020. The force-law and planetary examples
  discussed for the tutorials need exactly this last step.

## 3. Why it is hard

- **Trivial solutions.** $r \equiv 0$ fits every dataset, and so does $r = c \cdot s$ for any $s$
  that vanishes on it. Explicit regression avoids this by fixing $y$; implicit regression needs
  a normalisation. The nullspace formulation normalises the coefficient vector; Schmidt and
  Lipson match ratios of partial derivatives instead; SINDy-PI tries each library term as the
  left-hand side in turn.
- **Search cost.** Genetic search over expression trees is slow, and the implicit version has
  no target column to guide it.
- **Real locus.** A formula can vanish on the data and on extra spurious components, the same
  problem as [[Open Problems in Algebraic Implicit Learning]] §1.

## 4. What exists to build on

- **SymbolicRegression.jl** (and PySR on top of it): evolutionary search, Pareto fronts,
  custom losses, dimensional constraints, templates that fix part of a formula. Explicit
  regression; an implicit loss is possible through its custom-loss interface.
- **DataDrivenDiffEq.jl** (SciML): SINDy and an implicit optimiser for fixed libraries.
- This family's own nullspace fitting, which is implicit and exact for polynomial libraries.

A first experiment: recover the circle, then the robot arm's kinematic relation, from
samples, with an implicit loss in SymbolicRegression.jl, and compare with the polynomial
nullspace fit.

Related: [[Algebraic Implicit Learners]], [[Fitting is a Nullspace Problem]],
[[Open Problems in Algebraic Implicit Learning]], [[Geometric Deep Learning and Physical Laws]]
