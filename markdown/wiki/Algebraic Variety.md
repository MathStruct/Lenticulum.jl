#definition #algebraic-geometry #category-theory #scheme #computable

An **algebraic variety** is a geometric object defined as the [[zero set]] of a system of [[polynomial]] equations, providing the fundamental objects of study in [[algebraic geometry]] and forming the classical foundation that leads to the modern theory of [[scheme|schemes]].

## Classical Definition

### Affine Variety

An **affine algebraic variety** over an [[algebraically closed field]] $k$ is a subset $V \subseteq k^n$ of the form: $$V = \mathbf{V}(f_1, \ldots, f_m) = {(a_1, \ldots, a_n) \in k^n : f_i(a_1, \ldots, a_n) = 0 \text{ for all } i}$$

where $f_1, \ldots, f_m \in k[x_1, \ldots, x_n]$ are [[polynomial|polynomials]].

### Projective Variety

A **projective algebraic variety** is a subset $V \subseteq \mathbb{P}^n(k)$ defined by homogeneous polynomial equations, where $\mathbb{P}^n(k)$ is [[projective space]] over $k$.

## Zariski Topology

The **[[Zariski topology]]** on $k^n$ has:

- **Closed sets**: Algebraic varieties (zero sets of polynomial ideals)
- **Open sets**: Complements of algebraic varieties
- **Basis**: [[Principal open set|Principal open sets]] $D(f) = {x : f(x) \neq 0}$

This topology is typically much coarser than the usual [[metric topology]].

## Ideal-Variety Correspondence

The **[[Hilbert Nullstellensatz]]** establishes a correspondence between algebraic and geometric objects:

### Vanishing Ideal

For a subset $S \subseteq k^n$: $$\mathbf{I}(S) = {f \in k[x_1, \ldots, x_n] : f(a) = 0 \text{ for all } a \in S}$$

### Zero Set

For an ideal $I \subseteq k[x_1, \ldots, x_n]$: $$\mathbf{V}(I) = {a \in k^n : f(a) = 0 \text{ for all } f \in I}$$

### Nullstellensatz

Over an [[algebraically closed field]], there is a bijection: $${\text{radical ideals in } k[x_1, \ldots, x_n]} \leftrightarrow {\text{algebraic varieties in } k^n}$$

## Morphisms of Varieties

### Regular Maps

A **[[regular map]]** between varieties $\phi: V \to W$ is given by [[rational function|rational functions]] that are defined everywhere on $V$: $$\phi(x) = (f_1(x)/g_1(x), \ldots, f_m(x)/g_m(x))$$

where each $f_i/g_i$ is defined on all of $V$.

### Rational Maps

A **[[rational map]]** $\phi: V \dashrightarrow W$ is defined on a [[dense]] [[open subset]] of $V$ and given by [[rational function|rational functions]].

## Category of Varieties

Algebraic varieties form a [[category]] $\mathbf{Var}_k$ where:

- **Objects**: Algebraic varieties over $k$
- **[[Morphism|Morphisms]]**: [[Regular map|Regular maps]] between varieties
- **[[Composition]]**: Composition of regular maps
- **[[Identity morphism|Identity]]**: Identity map on each variety

## Coordinate Ring

### Affine Case

For an [[affine variety]] $V$, the **[[coordinate ring]]** is: $$k[V] = k[x_1, \ldots, x_n]/\mathbf{I}(V)$$

This gives a [[contravariant functor]]: $$k[-]: \mathbf{AffVar}_k^{\text{op}} \to \mathbf{Alg}_k$$

### Function Field

The **[[function field]]** $k(V)$ is the [[field of fractions]] of $k[V]$, consisting of [[rational function|rational functions]] on $V$.

## Examples

### Linear Varieties

- **Point**: $V = {(a_1, \ldots, a_n)}$ defined by $x_i - a_i = 0$
- **Line**: $V = {(t, at + b) : t \in k}$ in $k^2$
- **Hyperplane**: $a_1x_1 + \cdots + a_nx_n = 0$ in $k^n$

### Quadric Varieties

- **Circle**: $x^2 + y^2 = 1$ in $k^2$
- **Ellipse**: $\frac{x^2}{a^2} + \frac{y^2}{b^2} = 1$
- **Hyperbola**: $xy = 1$

### Cubic Curves

- **Elliptic curve**: $y^2 = x^3 + ax + b$ (if [[discriminant]] $\neq 0$)
- **Nodal cubic**: $y^2 = x^3 + x^2$ (with [[singular point]])

## Dimension Theory

### Krull Dimension

The **[[dimension]]** of a variety $V$ is the [[Krull dimension]] of its [[coordinate ring]] $k[V]$: $$\dim V = \dim k[V] = \max{\text{length of chains of prime ideals}}$$

### Transcendence Degree

Equivalently, $\dim V$ equals the [[transcendence degree]] of the [[function field]] $k(V)$ over $k$.

## Smoothness and Singularities

### Smooth Points

A point $p \in V$ is **[[smooth point|smooth]]** if the [[Jacobian matrix]] of defining equations has maximal rank at $p$.

### Singular Points

A **[[singular point]]** is where the variety fails to be smooth, indicating geometric pathology like [[cusp|cusps]], [[node|nodes]], or [[self-intersection|self-intersections]].

### Nonsingular Varieties

A variety is **[[nonsingular variety|nonsingular]]** (or **smooth**) if all its points are smooth.

## Projective Varieties

### Homogeneous Coordinates

Points in [[projective space]] $\mathbb{P}^n$ are represented by [[homogeneous coordinates]] $[x_0 : x_1 : \cdots : x_n]$.

### Projective Nullstellensatz

The correspondence extends to projective varieties using [[homogeneous ideal|homogeneous ideals]].

### Completeness

Projective varieties are **[[complete variety|complete]]** (compact in the [[Zariski topology]]), analogous to [[compactness]] in [[topology]].

## Sheaf Theory on Varieties

### Structure Sheaf

The **[[structure sheaf]]** $\mathcal{O}_V$ assigns to each open set $U \subseteq V$ the ring of [[regular function|regular functions]] on $U$.

### Coherent Sheaves

[[Coherent sheaf|Coherent sheaves]] on varieties generalize [[vector bundle|vector bundles]] and provide the foundation for [[algebraic geometry]].

## Birational Geometry

### Birational Equivalence

Two varieties are **[[birational equivalence|birationally equivalent]]** if they have isomorphic [[function field|function fields]], indicating they are "the same" outside lower-dimensional subsets.

### Resolution of Singularities

**[[Resolution of singularities]]** replaces a singular variety with a [[smooth variety|smooth]] one that is birationally equivalent.

## Connection to Schemes

### Motivation for Schemes

Classical varieties have limitations:

- Require [[algebraically closed field|algebraically closed fields]]
- Miss important geometric information
- Don't handle [[nilpotent element|nilpotents]] or [[non-reduced]] structures

### Scheme Theory

[[Scheme|Schemes]] generalize varieties by:

- Working over arbitrary [[base ring|base rings]]
- Including [[nilpotent element|nilpotent]] elements
- Providing better [[functorial]] properties
- Enabling [[relative geometry]]

## Categorical Properties

### Opposite Category

There is a [[contravariant functor|contravariant]] [[equivalence of categories|equivalence]]: $$\mathbf{AffVar}_k^{\text{op}} \simeq \mathbf{FinGenAlg}_k$$

between affine varieties and finitely generated $k$-[[algebra|algebras]].

### Functor of Points

The **[[functor of points]]** approach represents varieties as [[functor|functors]]: $$V: \mathbf{Alg}_k \to \mathbf{Set}$$ $$V(R) = {R\text{-points of } V}$$

## Modern Developments

### Derived Algebraic Geometry

[[Derived algebraic geometry]] extends varieties using [[derived category|derived categories]] and [[∞-category|∞-categories]].

### Tropical Geometry

[[Tropical geometry]] studies [[tropical variety|tropical varieties]] over the "[[tropical semiring]]" $(\mathbb{R} \cup {-\infty}, \oplus, \otimes)$.

### Arithmetic Geometry

[[Arithmetic geometry]] studies varieties over [[number field|number fields]] and their [[reduction]] modulo primes.

## Applications

### Cryptography

[[Elliptic curve]] varieties provide the foundation for [[elliptic curve cryptography]].

### Coding Theory

[[Algebraic coding theory]] uses varieties to construct [[error-correcting code|error-correcting codes]].

### Computer Vision

[[Computer vision]] algorithms use [[algebraic geometry]] for [[3D reconstruction]] and [[camera calibration]].

### Robotics

[[Robot]] motion planning uses [[configuration space|configuration spaces]] that are often algebraic varieties.

Algebraic varieties provide the classical foundation for [[algebraic geometry]], connecting [[algebra]] and [[geometry]] through the powerful machinery of [[polynomial]] equations and forming the conceptual basis for the modern theory of [[scheme|schemes]] in [[category theory]].