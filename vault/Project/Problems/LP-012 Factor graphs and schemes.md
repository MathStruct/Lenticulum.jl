#open-problem

> **Status: resolved within the agreed theory-note scope (2026-10-27; maintainer decision 2026-10-26). Original question (2026-10-22, Daniel: "Also is there a connection to schemes from algebraic geometry? I do not know much about that.").** Related: [[Factor Graphs and Schemes]], [[Algebraic Statistics Bridge]], [[Varieties Ideals and Real Nullstellensatz]], [[Branches and the Discriminant]], [[Composition is Elimination]], [[Bethe Free Energy]].

**Question.** Is there a connection between Lenticulum's factor graphs and schemes?

**What is already in the vault.** [[Algebraic Statistics Bridge]] shows that conditional independence is a polynomial (determinantal) constraint, so a discrete or Gaussian graphical model is an algebraic **variety**, and marginalisation is elimination. Schemes generalise varieties, so the question is what the generalisation adds. Three candidate links, none worked out here:

1. **Singular models.** Where a model's variety is singular, parameters are not identifiable and the Fisher metric degenerates; that is where model selection gets hard. Watanabe's singular learning theory (*Algebraic Geometry and Statistical Learning Theory*, 2009) handles this with resolution of singularities, and the real log canonical threshold replaces the parameter count in the evidence. Lenticulum's evidence and Bethe-criterion comparisons ([[Dynamic Graphs]] §4) would meet exactly this when learned factors are overparametrised.
2. **Tangent vectors are dual-number points.** In scheme language the tangent space of $X$ at a point is the set of maps $\mathrm{Spec}\,\mathbb R[\varepsilon]/(\varepsilon^2) \to X$: solutions of the defining equations over the dual numbers. That is forward-mode automatic differentiation, and the "jets" of the CT-ML wiki's [Information Geometry of Probabilistic Programs](https://mathstruct.org/CategoryTheory-ML-Wiki/Information-Geometry-of-Probabilistic-Programs). Schemes keep this first-order information even where the variety is singular, which is where the Fisher metric of [[Scalar and Multivariate Energy]] §6 is degenerate.
3. **Sheaves.** A scheme is built by gluing local pieces with a sheaf of functions. A factor graph can be read as a **cellular sheaf**: every variable and factor carries a space of local states, every edge a map saying how they must agree; global sections are the consistent assignments, and the sheaf Laplacian measures inconsistency (Hansen & Ghrist, *Toward a spectral theory of cellular sheaves*, 2019). Message passing would then be a way to compute (approximate) global sections. Sheaves are the ingredient factor graphs and schemes share.

**Leaning.** Link 3 is the closest to the code and would give the CT-ML wiki a *Cellular Sheaf* note; link 1 is the one with consequences for model selection. Both need reading before a vault note; the references above are to be checked and added to `refs.bib` first.

## 2026-10-27 — three links, with their limits

[[Factor Graphs and Schemes]] works out the three links with exact source definition/theorem
numbers and elementary proofs: Watanabe's parameter-evidence expansion and an RLCT example;
dual-number points and the Jacobian kernel (including a nonreduced point); and a cellular
sheaf for homogeneous linear factor relations, whose Laplacian kernel is exactly the
space of consistent state assignments. It links the toric/Plücker implementation notes,
[[L-SPEC-004 Algebraic Factors Toric and Grassmannian]] and [[Finding a Factor Graph]].

Qualifications to the original questions above: singular equation loci, nonidentifiability
and Fisher degeneracy are not equivalent conditions; the note gives an injective model
with a degenerate Fisher matrix. Watanabe's evidence integrates parameters, whereas the
worked Bethe example integrates states at fitted parameters. A cellular vector-space
sheaf does not supply a scheme's sheaf of local rings. The proved state-section
correspondence applies to linear hard constraints, not general probabilistic beliefs.
Identifying general BP with section computation remains a conjectural extension.

The theory note fulfils the agreed scope; no scheme objects, RLCT estimator, sheaf solver
or general nonlinear sheaf construction is implemented. The CT-ML wiki is not edited.
