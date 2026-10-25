#implementation #derivation

> `ExponentTable` enumerates affine or degree-exact polynomial indices. Monomials and
> tensor products of Chebyshev polynomials have exact polynomial derivatives; affine
> tables have an exact invertible change of basis.

> Sources: original implementation and polynomial recurrence derivations; code: `veronese.jl`, `../test/polynomial.jl` (Polynomial basis tables).
>
> Theory (CT-ML wiki): [Reverse Derivative Category](https://mathstruct.org/CategoryTheory-ML-Wiki/Reverse-Derivative-Category)

Columns of `exponents` are graded by increasing total degree, then descending
lexicographic order. For two coordinates and degree two the order is
`1,x,y,x²,xy,y²`. The matrix has N rows and binomial(N+d,d) columns in the affine
case, or binomial(N+d-1,d) in the degree-exact case. N is positive, d nonnegative.
There is no implicit data scaling or enforced domain.

For monomials the implementation uses powers built by multiplication and
products with one differentiated factor. It never divides by a coordinate:
zeros are regular inputs. Integers and rationals retain their arithmetic;
fixed-width integer overflow is ordinary Julia overflow, not arbitrary precision.
Float32 stays Float32. These are polynomial derivatives evaluated in the input
arithmetic, not a claim of exact floating-point results.

For Chebyshev, univariate values obey T₀=1, T₁=x and

$$
T_{n+1}=2xT_n-T_{n-1},\qquad T'_n=nU_{n-1},\qquad
U_0=1,\quad U_1=2x,\quad U_{n+1}=2xU_n-U_{n-1}.
$$

The multivariate value is the product of the univariate T terms. Products of
these values/derivatives avoid divisions at zeros or endpoints. Recurrences avoid
`acos` in evaluation. The intended conditioning benefit assumes data scaled to
[-1,1]; outside that interval values can grow rapidly. There is no guarantee that
this removes ill-conditioning for arbitrary sampling distributions.

`basis_change(mon,cheb)` expands each T by the integer recurrence and takes
products of coefficient vectors, returning a BigInt matrix C. On affine tables
it is triangular in the documented order with nonzero diagonal, hence invertible:

$$
v_{\mathrm{cheb}}(x)=C v_{\mathrm{mon}}(x),\qquad
\Theta_{\mathrm{mon}}=\Theta_{\mathrm{cheb}}C.
$$

A difficulty hidden by the phrase “homogeneous Chebyshev basis” is that
T₂(x)=2x²−1 is not homogeneous. `homogeneous=true` restricts **indices**, not the
polynomial support. A degree-exact Chebyshev table generally does not span the
homogeneous monomials, so a square C with this identity does not exist. The
conversion explicitly rejects missing lower terms instead of dropping them.
Degree zero and one conversions remain valid.

`Polynomial basis tables` tests counts and ordering through N=4,d=4, hand-written
monomials, integer/rational/Float32 evaluation, Chebyshev against cos(n acos x),
central differences (including endpoints), exact rational basis identities and
conversion rank through degree six in three variables. It does not test GPU,
high-degree stability, sparse storage or higher derivatives.
