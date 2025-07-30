Cat #definition #category-theory #2-category #computable #catlab

**Cat** is the [[2-category]] of all small [[category|categories]], where [[Object|objects]] are small categories, [[1-morphism|1-morphisms]] are [[functor|functors]], and [[2-morphism|2-morphisms]] are [[natural transformation|natural transformations]].

## Formal Definition

The 2-category **Cat** consists of:

### 0-cells (Objects)

Small [[category|categories]] $\mathcal{C}, \mathcal{D}, \mathcal{E}, \ldots$

### 1-cells (1-morphisms)

[[Functor|Functors]] $F: \mathcal{C} \to \mathcal{D}$ between categories

### 2-cells (2-morphisms)

[[Natural transformation|Natural transformations]] $\alpha: F \Rightarrow G$ between functors $F, G: \mathcal{C} \to \mathcal{D}$

## Composition Structure

### Horizontal Composition (1-morphisms)

For functors $F: \mathcal{C} \to \mathcal{D}$ and $G: \mathcal{D} \to \mathcal{E}$: $$G \circ F: \mathcal{C} \to \mathcal{E}$$

### Vertical Composition (2-morphisms)

For natural transformations $\alpha: F \Rightarrow G$ and $\beta: G \Rightarrow H$: $$\beta \cdot \alpha: F \Rightarrow H$$

### Horizontal Composition (2-morphisms)

For natural transformations $\alpha: F \Rightarrow G: \mathcal{C} \to \mathcal{D}$ and $\beta: H \Rightarrow K: \mathcal{D} \to \mathcal{E}$: $$\beta \circ \alpha: H \circ F \Rightarrow K \circ G$$

## Key Properties

### Identity Elements

- **Identity functor**: $\text{Id}_{\mathcal{C}}: \mathcal{C} \to \mathcal{C}$ for each category $\mathcal{C}$
- **Identity natural transformation**: $\text{id}_F: F \Rightarrow F$ for each functor $F$

### Interchange Law

The fundamental coherence condition: $$(\beta_2 \cdot \alpha_2) \circ (\beta_1 \cdot \alpha_1) = (\beta_2 \circ \beta_1) \cdot (\alpha_2 \circ \alpha_1)$$

## Important Subcategories

### Cat₁ (1-Category Structure)

When viewed as a [[1-category]], **Cat** has:

- Objects: Small categories
- Morphisms: Functors
- Natural transformations become equalities of functors

### Groupoids

The [[full subcategory]] **Gpd** ⊆ **Cat** of small [[groupoid|groupoids]]

## Cartesian Structure

**Cat** is a [[cartesian monoidal category]] with:

- **Product**: $\mathcal{C} \times \mathcal{D}$ (product category)
- **Terminal object**: $\mathbf{1}$ (the terminal category with one object and one morphism)

<!-- \begin{tikzcd} \mathcal{C} \times \mathcal{D} & \mathcal{C} \arrow[l, "\pi_1"'] \\ & \mathcal{D} \arrow[ul, "\pi_2"] \end{tikzcd} -->

## Limits and Colimits

**Cat** has many [[limit|limits]] and [[colimit|colimits]]:

- **[[Pullback|Pullbacks]]**: [[Comma category|Comma categories]]
- **[[Pushout|Pushouts]]**: Certain [[amalgamated sum|amalgamated sums]]
- **[[Equalizer|Equalizers]]**: Categories of natural transformations

## Size Issues

### Large Categories

For [[large category|large categories]], we consider:

- **CAT**: The 2-category of all categories (requires careful foundational treatment)
- **Cat**: Often used for locally small categories in some contexts

## Applications

### Topos Theory

**Cat** provides the context for [[elementary topos|elementary toposes]] and [[Grothendieck topos|Grothendieck toposes]]

### Homotopy Theory

The [[homotopy category]] **Ho(Cat)** and model structures on **Cat**

### Higher Category Theory

**Cat** embeds into [[n-category|higher categories]] like [[bicategory|bicategories]] and [[tricategory|tricategories]]

### Computational Applications

In [[categorical semantics]], **Cat** models:

- [[Type theory]] and programming languages
- [[Database theory]] and query languages
- [[Concurrency theory]] and process calculi

**Cat** serves as the fundamental 2-category for studying [[category theory|categorical]] structures and their relationships through [[functor|functors]] and [[natural transformation|natural transformations]].