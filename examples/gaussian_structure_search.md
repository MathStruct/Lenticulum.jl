# Gaussian structure search (LP-010)

Run `julia --project=. examples/gaussian_structure_search.jl` from the repository root.
The published explanation and measured trajectory are in
[`Finding a Factor Graph`](../vault/Factor%20Graphs/Finding%20a%20Factor%20Graph.md).
The script is included by `test/structure_search.jl` without running `main()`.

This is an example of the top-level package, since Lenticulum owns `GaussianFactor` and
`GaussianPrior` and Mycelium owns the graph/BP/Bethe operations. No package source or
public API changes are needed. Newton precision fitting and graphical-lasso ADMM are
small dense reference solvers, kept local to the example rather than installed as a
general structure-learning facility.

The pipeline fixes the mean at zero, fits every candidate's allowed precision entries
by exact Gaussian maximum likelihood, represents that precision by unary/pairwise
Gaussian potentials, then estimates the normalised data NLL using Mycelium's Bethe
normaliser. The optional BIC term counts d diagonal and one parameter per edge. Potential
normalisers appear in both data energy and the unclamped free energy and cancel. BP
operates on fresh graphs; it is not differentiated through and is not used as the fitting
optimiser. A proposed edge causes all allowed parameters and unary remainders to be refitted.

The factorisation requires strict diagonal dominance and stops with an error if that
condition fails; it does not handle arbitrary positive-definite precision matrices.
The fixture is deliberately in that class. Newton convergence and BP convergence are
checked. A small objective-roundoff allowance in the Armijo test avoids stalling when
the objective has reached machine precision while its gradient still needs a Newton
step. Unary messages and cavities are seeded explicitly before synchronous BP.

Graphical lasso uses the off-diagonal penalty convention and ADMM with rho=1, following
Boyd et al. (2011) §6.5, eq. (6.5). The precision proximal step is a symmetric eigensolve;
the split variable has exact off-diagonal zeros after soft thresholding. Both primal
and dual residuals must converge. A fixed logarithmic penalty path proposes supports;
exact refitted BIC selects among them. Labels are only read by the final precision/recall
report. The test independently checks KKT conditions and the unpenalised inverse-covariance
solution, in addition to actual chain recovery and forest normaliser comparisons.

The main difficulty is statistical rather than computational: fitted likelihood has no
automatic parameter-count penalty, and loopy Bethe error can dominate the BIC edge cost.
The default eight-variable run retains one false edge even with BIC; the stronger-coupling
stress case retains three. The separate six-variable regression fixture recovers its chain.
All are reported by the vault note; no consistency or general recovery guarantee is claimed.
There is no deletion search, held-out tuning, non-Gaussian fitting, TDA implementation,
joint-factor splitting, or parameter evidence integration.
