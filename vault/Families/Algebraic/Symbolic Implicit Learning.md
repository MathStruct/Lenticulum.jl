#open-problem

> Learn a relation $r(z) = 0$ as a **formula**: not a polynomial with fixed monomials, not a
> network, but a short symbolic expression found by search. It would make learned relations
> readable and comparable, and it is slow and hard. Noted for later; nothing is implemented. The
> most promising route is distillation from an already trained factor (§5).

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
  afterwards by a formula, as in Cranmer et al. 2020, and the formula then corrected by a
  learned residual. This is the route that avoids most of §3; see §5.

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

## 5. Distillation: a learned factor as the teacher

The trivial solution of §3 is a problem of learning **from samples alone**: samples say only
where $r$ vanishes. A trained factor says much more. Its field $g(z)$ is defined everywhere,
points towards the relation, and vanishes only on it; the solver projects any point onto the
relation; an energy network (or a closed-form mixture) even has an energy $U$. So the plan is
in three steps: **learn a factor, distil it into a formula, learn the residual.**

### 5.1 Distillation losses that exclude $r \equiv 0$

| teacher signal | loss for the formula $r$ | why $r \equiv 0$ fails |
|---|---|---|
| projections $p_i = \mathrm{proj}(z_i)$ of off-relation points (`implicit_infer` with $\rho = 0$) and normals $n_i = (z_i - p_i)/\lVert z_i - p_i\rVert$ | $r(p_i) \approx 0$, $\nabla r(p_i) \parallel n_i$, $\lVert\nabla r(p_i)\rVert = 1$ | the gradient must have unit length on the relation |
| the signed distance $d_i = \pm\lVert z_i - p_i\rVert$ near the relation | $r(z_i) / \lVert\nabla r(z_i)\rVert \approx d_i$ | a first-order distance estimate, scale-free |
| the field $g$ itself (any learner) | $\nabla E_{\mathrm{sym}}(z_i) \approx g(z_i)$, the relation being the minima of $E_{\mathrm{sym}}$ | $g \neq 0$ off the relation, so a constant $E$ does not fit |
| the energy $U$ (energy networks, mixtures) | $E_{\mathrm{sym}}(z_i) \approx U(z_i) + c$ | as above, with values instead of gradients |

The first is implicit-surface fitting with normals, a classical way to normalise; here the
normals come free from the teacher. The third and fourth fit a *density* rather than a
relation: cleaner as losses, but the target is the teacher's **smoothed** log-density, which is
not a short formula even for a circle (it has terms in $\log r$ and a smoothing width).
Fitting the zero set and normals gives the short formula ($x^2 + y^2 - 1$, normalised).

Two refinements:

- **Avoid inheriting the teacher's bias.** The teacher's relation sits slightly inside a
  curved one (the smoothing bias, [[Implicit Diffusion Learners]] §5). Fit the zero set to the
  **data**, and take only the normals, and the off-relation points, from the teacher.
- **Several branches.** A multivalued relation is one formula ($r = (y - f_1)(y - f_2)$ is
  allowed), so the implicit version needs no branch labels, unlike explicit symbolic
  regression on "y given x". `implicit_roots` supplies all branches for sampling targets.

### 5.2 The residual

Keep the formula and learn what it misses. The tool for this exists: a product of relations
adds fields ([[Composing Diffusion Factors]], `ProductRelation`), so

$$g_{\text{total}} = \beta_{\mathrm{sym}}\,\nabla E_{\mathrm{sym}} + g_{\text{res}},$$

with $g_{\text{res}}$ a small diffusion model trained on the data **with the symbolic factor
fixed**. Its denoising target is the part of the noise the formula does not explain,
$\varepsilon - \sigma_t\nabla E_{\mathrm{sym}}(x_t)$; this is exact only at small noise levels
(the composition obstruction T9 again), which are the levels the implicit learner reads. The
residual must be penalised towards zero, or it absorbs everything and the formula becomes
decoration; its size is then the honest measure of what the formula misses. Iterating (distil
the residual too) is the AI-Feynman or sparse-regression pattern of peeling off one term at a
time.

What one gets: a readable relation that is exact where the physics is known, a learned
correction where it is not, and the free energy (or the residual's norm) to decide whether a
term is worth adding. The force-law tutorial already did the explicit half of this
(symbolic regression on a learned pair force, as in Cranmer et al. 2020); this is the
implicit version.

### 5.3 What to try first

1. Teacher: the circle model of the training tutorial. Distil with the projection-and-normal
   loss in SymbolicRegression.jl (custom loss, its expression gradients), expect
   $x^2 + y^2 - 1$.
2. A relation with a known part and an unknown part, e.g. the robot arm with a sagging second
   link: the formula should find rigid kinematics, the residual the sag.
3. Check that the residual factor plus the formula answers queries better than either alone,
   and that the residual vanishes on data with no unknown part.

Related: [[Algebraic Implicit Learners]], [[Fitting is a Nullspace Problem]],
[[Open Problems in Algebraic Implicit Learning]], [[Geometric Deep Learning and Physical Laws]],
[[Composing Diffusion Factors]], [[Implicit Diffusion Learners]]
