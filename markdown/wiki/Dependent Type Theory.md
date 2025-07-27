#definition #type-theory #logic #category-theory #dependent-types #foundation

**Dependent type theory** is a formal system where [[type|types]] can depend on values, providing a foundation for mathematics that unifies [[logic]], [[computation]], and [[category theory]] through the [[propositions-as-types]] correspondence.

## Formal Definition

A **dependent type theory** consists of:

### Judgments

- **Type formation**: $\Gamma \vdash A : \mathsf{Type}$ (A is a type in context $\Gamma$)
- **Term formation**: $\Gamma \vdash a : A$ (a is a term of type A in context $\Gamma$)
- **Type equality**: $\Gamma \vdash A \equiv B : \mathsf{Type}$
- **Term equality**: $\Gamma \vdash a \equiv b : A$

### Contexts

A **context** $\Gamma$ is a sequence of variable declarations: $$\Gamma ::= \cdot \mid \Gamma, x : A$$

where each type $A$ may depend on previous variables in $\Gamma$.

## Dependent Function Types (Π-types)

The fundamental constructor is the **dependent product** or **Π-type**:

### Formation Rule

$$\frac{\Gamma \vdash A : \mathsf{Type} \quad \Gamma, x : A \vdash B : \mathsf{Type}}{\Gamma \vdash \Pi_{x:A} B : \mathsf{Type}}$$

### Introduction Rule

$$\frac{\Gamma, x : A \vdash b : B}{\Gamma \vdash \lambda x.b : \Pi_{x:A} B}$$

### Elimination Rule

$$\frac{\Gamma \vdash f : \Pi_{x:A} B \quad \Gamma \vdash a : A}{\Gamma \vdash f(a) : B[a/x]}$$

### Computation Rule

$$\Gamma \vdash (\lambda x.b)(a) \equiv b[a/x] : B[a/x]$$

## Dependent Pair Types (Σ-types)

The **dependent sum** or **Σ-type** represents existential quantification:

### Formation Rule

$$\frac{\Gamma \vdash A : \mathsf{Type} \quad \Gamma, x : A \vdash B : \mathsf{Type}}{\Gamma \vdash \Sigma_{x:A} B : \mathsf{Type}$$

### Introduction Rule

$$\frac{\Gamma \vdash a : A \quad \Gamma \vdash b : B[a/x]}{\Gamma \vdash (a, b) : \Sigma_{x:A} B}$$

### Elimination Rules

$$\frac{\Gamma \vdash p : \Sigma_{x:A} B}{\Gamma \vdash \pi_1(p) : A} \quad \frac{\Gamma \vdash p : \Sigma_{x:A} B}{\Gamma \vdash \pi_2(p) : B[\pi_1(p)/x]}$$

## Identity Types

**Identity types** capture [[equality]] as a first-class notion:

### Formation Rule

$$\frac{\Gamma \vdash A : \mathsf{Type} \quad \Gamma \vdash a, b : A}{\Gamma \vdash \mathsf{Id}_A(a, b) : \mathsf{Type}}$$

### Introduction Rule ([[Reflexivity]])

$$\frac{\Gamma \vdash a : A}{\Gamma \vdash \mathsf{refl}_a : \mathsf{Id}_A(a, a)}$$

### Elimination Rule ([[J-rule]])

$$\frac{\Gamma, x : A, y : A, p : \mathsf{Id}_A(x, y) \vdash C : \mathsf{Type} \quad \Gamma, z : A \vdash c : C[z, z, \mathsf{refl}_z/x, y, p]}{\Gamma \vdash J : \Pi_{x,y:A} \Pi_{p:\mathsf{Id}_A(x,y)} C}$$

## Inductive Types

**Inductive types** are defined by their constructors and [[induction]] principles:

### Natural Numbers

$$\frac{}{\Gamma \vdash \mathbb{N} : \mathsf{Type}} \quad \frac{}{\Gamma \vdash 0 : \mathbb{N}} \quad \frac{\Gamma \vdash n : \mathbb{N}}{\Gamma \vdash S(n) : \mathbb{N}}$$

### Induction Principle

$$\frac{\Gamma, n : \mathbb{N} \vdash P : \mathsf{Type} \quad \Gamma \vdash p_0 : P[0/n] \quad \Gamma, m : \mathbb{N}, p : P[m/n] \vdash p_S : P[S(m)/n]}{\Gamma \vdash \mathsf{ind}_{\mathbb{N}} : \Pi_{n:\mathbb{N}} P}$$

## Universe Hierarchy

To avoid [[paradox|paradoxes]], types are stratified into [[universe|universes]]: $$\mathsf{Type}_0 : \mathsf{Type}_1 : \mathsf{Type}_2 : \cdots$$

### Universe Rules

$$\frac{\Gamma \vdash A : \mathsf{Type}_i}{\Gamma \vdash A : \mathsf{Type}_{i+1}} \quad \frac{\Gamma \vdash A : \mathsf{Type}_i \quad \Gamma, x : A \vdash B : \mathsf{Type}_j}{\Gamma \vdash \Pi_{x:A} B : \mathsf{Type}_{\max(i,j)}}$$

## Categorical Semantics

### Locally Cartesian Closed Categories

Dependent type theory is modeled by [[locally cartesian closed category|locally cartesian closed categories]] (LCCCs):

- Types: Objects in slice categories $\mathcal{C}/\Gamma$
- Terms: Sections of projections
- Dependent products: Right adjoints to pullback
- Dependent sums: Left adjoints to pullback

### Interpretation Diagram

<!-- \begin{tikzcd} \mathcal{C}/(\Gamma, x:A) \arrow[r, "\Pi_x"] \arrow[d, "p^*"'] & \mathcal{C}/\Gamma \arrow[d, "q^*"] \\ \mathcal{C}/\Gamma \arrow[r, "\Sigma_x"'] & \mathcal{C}/\Delta \end{tikzcd} -->

## Propositions as Types

The **[[propositions-as-types]]** correspondence identifies:

- **Propositions** ↔ **Types**
- **Proofs** ↔ **Terms**
- **Logical operations** ↔ **Type constructors**

### Logical Correspondence

- $A \land B$ ↔ $A \times B$ (product type)
- $A \lor B$ ↔ $A + B$ (sum type)
- $A \Rightarrow B$ ↔ $A \to B$ (function type)
- $\forall x:A. P(x)$ ↔ $\Pi_{x:A} P(x)$ (dependent product)
- $\exists x:A. P(x)$ ↔ $\Sigma_{x:A} P(x)$ (dependent sum)

## Examples

### Vector Type

A type of vectors parameterized by length: $$\mathsf{Vec} : \mathbb{N} \to \mathsf{Type}$$ $$\mathsf{nil} : \mathsf{Vec}(0)$$ $$\mathsf{cons} : \Pi_{n:\mathbb{N}} A \to \mathsf{Vec}(n) \to \mathsf{Vec}(S(n))$$

### Matrix Multiplication

Type-safe matrix multiplication: $$\mathsf{mult} : \Pi_{m,n,p:\mathbb{N}} \mathsf{Matrix}(m,n) \to \mathsf{Matrix}(n,p) \to \mathsf{Matrix}(m,p)$$

### Sorted Lists

Lists that are provably sorted: $$\mathsf{SortedList}(A) = \Sigma_{xs:\mathsf{List}(A)} \mathsf{IsSorted}(xs)$$

## Homotopy Type Theory

**[[Homotopy type theory]]** (HoTT) extends dependent type theory with:

- **[[Univalence axiom]]**: $(A \equiv B) \simeq (A \cong B)$
- **[[Higher inductive type|Higher inductive types]]**: Types with path constructors
- **[[∞-groupoid]]** interpretation of types

### Path Induction

Identity types are interpreted as [[path|paths]] in [[topological space|topological spaces]], making types into [[∞-groupoid|∞-groupoids]].

## Computational Content

### Extraction

Proofs in dependent type theory can be [[program extraction|extracted]] to functional programs, providing a foundation for [[certified programming]].

### Normalization

[[Strong normalization]] ensures that all well-typed terms have [[normal form|normal forms]], making [[type checking]] decidable.

## Applications

### Proof Assistants

- **[[Coq]]**: Based on [[Calculus of Inductive Constructions]]
- **[[Agda]]**: Pure dependent type theory with pattern matching
- **[[Lean]]**: Modern proof assistant with powerful [[automation]]

### Certified Programming

- **[[Software verification]]**: Proving program correctness
- **[[Security]]**: Cryptographic protocol verification
- **[[Safety-critical systems]]**: Aerospace and medical software

### Mathematics Formalization

- **[[Univalent Foundations]]**: New foundations for mathematics
- **[[Formal mathematics]]**: Computer-verified mathematical proofs
- **[[Mathematical libraries]]**: Formalized mathematical theories

## Type-Theoretic Foundations

### Intensional vs Extensional

- **Intensional**: Identity requires proof (standard in proof assistants)
- **Extensional**: Propositional equality implies definitional equality

### Predicative vs Impredicative

- **Predicative**: Universe stratification prevents [[impredicativity]]
- **Impredicative**: Allows quantification over all propositions

Dependent type theory provides a rich foundation that unifies [[logic]], [[computation]], and [[mathematics]], offering both theoretical insights through [[category theory]] and practical applications in [[formal verification]] and [[certified programming]].