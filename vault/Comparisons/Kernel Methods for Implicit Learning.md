#comparison #open-problem

> Can kernel methods learn relations? Plain kernel regression cannot: asked for a function
> that vanishes on the data, it returns the zero function. Every kernel method that does learn
> a relation adds something that rules out that trivial solution, and there are several
> classical ones. One of them is already in this project: the closed-form circle model of the
> first tutorial is a Gaussian kernel density estimate, and its relation is the density ridge.

> Sources: measurements from `lib/VariationalDiffusion.jl` (`kde_predictor`, `kde_bandwidth`) and the kernel-baseline tutorial; Carr et al., SIGGRAPH 2001; Turk & O'Brien, ACM TOG 2002; Macêdo, Gois & Velho, Computer Graphics Forum 2011; Williams & Fitzgibbon, *Gaussian Process Implicit Surfaces*, 2006; Hoffmann, Pattern Recognition 2007; Schölkopf et al., Neural Computation 2001; Tax & Duin, Machine Learning 2004; Genovese et al., Annals of Statistics 2014; Ozertem & Erdogmus, JMLR 2011; Sriperumbudur et al., JMLR 2017; Livni et al., ICML 2013; full entries in [[Bibliography]]
>
> Theory (CT-ML wiki): [Statistical Game](https://mathstruct.org/CategoryTheory-ML-Wiki/Statistical-Game)

## 1. Why plain kernel regression gives no zeros

Fit $f(z) = \sum_i \alpha_i\,k(z_i, z)$ so that $f(z_i) = 0$ on the data. That is $K\alpha = 0$ with the
kernel matrix $K$, and for a strictly positive definite kernel (the Gaussian RBF, for example)
the only solution is $\alpha = 0$: $f \equiv 0$. This is the **trivial solution** that every implicit
learner has to exclude ([[Implicit Learners]] §2), in its sharpest form. The methods below
differ in *how* they exclude it.

## 2. The methods

### Implicit surfaces: constraints off the relation

**Carr et al. (2001)** reconstruct 3-D shapes with radial basis functions: $f = 0$ at surface
points, plus $f = \pm d$ at points displaced off the surface along the normals. The extra
constraints make $f$ an approximate signed distance, and its zero set is the surface.
**Turk & O'Brien (2002)** do the same in their variational implicit surfaces; **Hermite RBF
implicits** (Macêdo, Gois & Velho, 2011) use normal (gradient) constraints instead of extra
points. This is kernel implicit learning with labelled data, standard in geometry processing
for twenty-five years.

### Gaussian process implicit surfaces: a belief over the relation

**Williams & Fitzgibbon (2006)** put a Gaussian process on $f$ with the same kind of
constraints. The posterior is a distribution over functions, hence over their zero sets: where
the relation is, *and how sure the model is*. Robotics uses this for grasping and mapping from
sparse touch or range data. For this project it is the interesting one: point inference in the
other families returns a Dirac belief ([[Open Problems in Implicit Diffusion Learning]] T1, I2),
and a GP implicit factor would return a belief over the relation itself.

### Kernel PCA: a residual from the data alone

Map the data into feature space with $\varphi$ and project onto the leading principal
components. The reconstruction error $r(z) = \lVert \varphi(z) - P\varphi(z)\rVert^2$, computable with
kernels only, is small on the data manifold and grows away from it; **Hoffmann (2007)** uses it
for novelty detection. No labels and no off-surface points are needed: the eigenproblem's
normalisation excludes the trivial solution, just as the coefficient normalisation does in the
algebraic family ([[Fitting is a Nullspace Problem]]).

### Support estimation: a region, not a manifold

The **one-class SVM** (Schölkopf et al., 2001) and **support vector data description** (Tax &
Duin, 2004) learn $f$ whose zero level set encloses the data. That answers "is this
configuration admissible?", but the relation is a full-dimensional region, so there is no
root to find inside it; it is a membership test rather than an implicit learner in this vault's
sense.

### Density ridges: the kernel version of the diffusion family

A Gaussian kernel density estimate is a Gaussian mixture with one component per data point.
The ridge of a density (points where it is maximal in the directions across the relation) is a
principal curve or surface; estimating it is **nonparametric ridge estimation**
(Genovese et al., 2014), and **subspace-constrained mean shift** (Ozertem & Erdogmus, 2011)
finds ridge points iteratively.

> [!important] This project already has one
> `GaussianMixtureEps` with 48 centres of width 0.05 on the unit circle, the closed-form model
> of the first tutorial and the README, *is* a Gaussian kernel density estimate. The implicit
> learner's stable roots for it are the ridge of the KDE smoothed at the field nodes' noise
> levels ([[Implicit Diffusion Learners]] §5). A diffusion network replaces the kernel sum by a
> network that scales with the data; the kernel version is the exact, small-data baseline.

### Kernel energies: score matching in an RKHS

**Infinite-dimensional kernel exponential families** (Sriperumbudur et al., 2017) model
$\log p(z) = f(z) - \log Z$ with $f$ in a reproducing kernel Hilbert space and fit $f$ by score
matching, which has a closed-form solution there. The score $\nabla f$ is a gradient by
construction, so this is the kernel counterpart of the energy-parametrised diffusion models
([[energy]]), with convex fitting in place of training.

### Vanishing ideals: the algebraic family with data-driven bases

**Vanishing Component Analysis** (Livni et al., 2013) constructs polynomials that
approximately vanish on the data, the algebraic family's fitting problem, with a basis built
from the data rather than fixed in advance ([[Algebraic Implicit Learners]]).

## 3. Summary

| method | residual | trivial solution excluded by | the relation is | uncertainty | closest family here |
|---|---|---|---|---|---|
| RBF implicit surfaces | $f$ in an RBF span | off-surface or gradient constraints | the zero set | none | algebraic (fixed basis) |
| GP implicit surfaces | $f \sim \mathcal{GP}$ | the same constraints, as observations | the zero set | **posterior over the zero set** | none yet |
| kernel PCA | reconstruction error in feature space | eigenvector normalisation | a low-error valley | none | algebraic (nullspace fit) |
| one-class SVM, SVDD | decision function | the margin / volume term | a region | none | not a relation in this sense |
| KDE ridge, SCMS | the KDE's gradient across the ridge | none needed: the density has ridges | the density ridge | via the density | **diffusion** (`GaussianMixtureEps`) |
| kernel exponential family | RKHS log-density | normalisation of the density | the density ridge | via the density | diffusion with an energy ([[energy]]) |
| VCA | data-driven polynomials | normalisation of the coefficients | the zero set | none | algebraic |

## 4. What kernels would offer, and what they cost

**Offer:** closed-form or convex fitting, well-understood statistics, uncertainty for free in
the GP case, and excellent behaviour in low dimensions, which is exactly the regime of a single
relation over a few coordinates.

**Cost:** the kernel matrix makes fitting cubic in the number of data points (inducing points
and random features reduce this), and kernels with fixed bandwidths behave poorly in high
dimensions.

**For Lenticulum**, two concrete uses:
1. a **kernel baseline** for the diffusion family, **now built and measured**. `kde_predictor`
   turns samples into a KDE noise predictor (the closed-form mixture, so queries are exact) and
   `kde_bandwidth` picks the bandwidth by held-out likelihood. On the robot arm, with the same
   fixed data, the diffusion network wins clearly where data are scarce:

   | training samples | KDE ridge: median miss | diffusion network: median miss | KDE s/query | network s/query |
   |---|---|---|---|---|
   | 500 | 0.080 | 0.010 | 0.16 | 0.09 |
   | 4000 | 0.019 | 0.009 | 0.48 | 0.08 |

   The configurations form a 2-D surface in 4-D; the KDE's answers snap towards nearby samples,
   so its error follows how densely they cover the surface, and tuning the bandwidth on the
   query errors themselves still leaves about 0.07 at 500 samples. The network interpolates the
   surface between samples. Worked through in the documentation's tutorial *What does the network
   add? A kernel baseline*.
2. a **GP implicit factor**, not built: the first factor whose answer is a belief over the relation rather
   than a point, which also needs the Gaussian-message machinery of [[Belief Algebra]] §6.

Related: [[Implicit Learners]], [[Implicit Diffusion Learners]], [[Algebraic Implicit Learners]],
[[Fitting is a Nullspace Problem]], [[energy]], [[Symbolic Implicit Learning]],
[[Open Problems in Implicit Diffusion Learning]], [[Belief Algebra]]
