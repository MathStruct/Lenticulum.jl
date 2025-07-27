# Mathematical Wiki Documentation

This is an Obsidian vault designed for mathematical knowledge representation, optimized for future integration with graph databases and vector embeddings for RAG (Retrieval-Augmented Generation) systems.

## Structure and Organization

### Folder Structure

```
markdown/
├── wiki/           # Core concept definitions
├── examples/       # Concrete examples and applications
├── theorems/       # Mathematical theorems and proofs
├── lemmas/         # Supporting lemmas
├── propositions/   # Mathematical propositions
├── corollaries/    # Corollaries to theorems
└── tikz_processor.jl
```

### File Organization Rules

**One Concept Per File**: Each markdown file must contain exactly one mathematical concept, theorem, or definition. This ensures maximum granularity for database indexing.

**No SVG Storage**: Never save markdown files in `wiki/SVGs/` or `examples/SVGs/`. These directories are reserved for generated diagram files only.

**Granular Content**: Make definitions and statements as compact and atomic as possible while maintaining mathematical precision.

## Content Guidelines

### Wiki Folder

The `wiki/` folder contains **definitions only**:

- Pure conceptual definitions
- No examples, applications, or illustrations
- Maximum linking to related concepts
- Minimal, precise mathematical statements

**Example Structure:**

```markdown
# Group

A **group** is a [[set]] $G$ equipped with a [[binary operation]] $\cdot: G \times G \to G$ satisfying:
- [[Associativity]]: $(a \cdot b) \cdot c = a \cdot (b \cdot c)$ for all $a,b,c \in G$
- [[Identity element]]: There exists $e \in G$ such that $e \cdot a = a \cdot e = a$ for all $a \in G$
- [[Inverse element]]: For each $a \in G$, there exists $a^{-1} \in G$ such that $a \cdot a^{-1} = a^{-1} \cdot a = e$
```

### Examples Folder

The `examples/` folder contains:

- Concrete instances of concepts
- Applications and use cases
- Computational examples
- Must link back to corresponding wiki definitions

**Naming Convention:** `concept-name-example.md`

### Linking Strategy

**Universal Linking**: Link every mathematical term using double square brackets `[[term]]`, even if the target file doesn't exist yet. Obsidian will create placeholder links that can be resolved later.

**Consistent Terminology**: Use the exact same term consistently across all files to ensure proper linking.

**Bidirectional References**: Leverage Obsidian's backlink system to create a rich connection graph.

**Implementation Links**: When available, include links to actual code implementations:

- Repository implementations: `[Implementation](https://github.com/repo/file.jl)`
- CatLab.jl references: `[CatLab.jl](https://github.com/AlgebraicJulia/Catlab.jl/blob/main/src/path/to/file.jl)`

### Hashtag Usage

**Primary Classification Tags** (choose one):

- `#definition` - Core mathematical definitions
- `#theorem` - Major theorems and results
- `#lemma` - Supporting lemmas
- `#proposition` - Mathematical propositions
- `#corollary` - Corollaries to theorems

**Mathematical Domain Tags** (multiple allowed):

- `#category-theory` - Categorical concepts
- `#algebra` - Algebraic structures
- `#topology` - Topological concepts
- `#logic` - Logical concepts
- `#set-theory` - Set-theoretic concepts

**Computational Tags** (when applicable):

- `#computable` - Has algorithmic implementation
- `#catlab` - Implemented in CatLab.jl
- `#constructive` - Constructively defined

**Example Usage:**

```markdown
# Group

#definition #algebra #computable

A **group** is a [[set]] $G$ equipped with...

[CatLab.jl Implementation](https://github.com/AlgebraicJulia/Catlab.jl/blob/main/src/theories/Group.jl)
```

## Writing Guidelines

### Compactness Principles

1. **Atomic Definitions**: Each concept should be irreducible - cannot be meaningfully split into smaller parts
2. **Essential Information Only**: Include only what's necessary to define the concept
3. **Maximum Cross-Referencing**: Link to every related concept mentioned
4. **Precise Language**: Use standard mathematical terminology consistently

### Prohibited Content in Wiki

- Examples or specific instances
- Historical context or motivation
- Computational procedures
- Extended explanations or intuition
- Proof sketches or derivations

### Database Optimization

**Why This Structure?**

This organization is specifically designed for:

- **Graph Database Integration**: Each file becomes a node with rich connection metadata
- **Vector Embeddings**: Compact, focused content produces more meaningful embeddings
- **RAG Systems**: Granular concepts enable precise retrieval and composition
- **Semantic Search**: Well-linked atomic concepts improve search relevance

## TikZ Diagrams

### Creating Diagrams

To include mathematical diagrams, place TikZ code inside HTML comments:

```markdown
<!--
\begin{tikzcd}
A \arrow[r, "f"] \arrow[d, "g"'] & B \arrow[d, "h"] \\
C \arrow[r, "k"'] & D
\end{tikzcd}
-->
```

### Processing Diagrams

Use the provided `tikz_processor.jl` script to automatically:

1. Extract TikZ code from HTML comments
2. Generate SVG images
3. Embed them in your markdown files

Run: `julia tikz_processor.jl` in your vault directory.

### Diagram Guidelines

- **Minimal Diagrams**: Include only essential structural information
- **No Examples**: Diagrams in wiki files should illustrate the abstract concept only
- **Consistent Style**: Use standard mathematical diagram conventions
- **TikZ-CD**: Prefer `tikz-cd` for categorical diagrams

## Quality Standards

### File Requirements

- **Title**: Must match the primary concept name
- **Definition**: Clear, concise mathematical definition
- **Links**: All related concepts must be linked
- **No Redundancy**: Avoid repeating information available in linked concepts
- **Mathematical Notation**: Use consistent LaTeX notation

### Cross-Reference Network

Build a dense network of mathematical relationships:

- Generalizations and specializations
- Prerequisites and dependencies
- Related concepts and analogies
- Categorical relationships

## Future Integration

This vault structure enables:

- Automated knowledge graph construction
- Semantic similarity calculations
- Dependency analysis
- Concept clustering and classification
- Intelligent mathematical reasoning systems

The granular, highly-linked structure ensures that when converted to a graph database with vector embeddings, the mathematical knowledge will be optimally represented for AI-powered mathematical assistance and reasoning.


## Note to the reader

I didn't even keep a percentage of what this documentation says. 
I just list some problems that I see:
- Someone should probably update nLab to include the Category theory + topic X stuff (machine learning, )
- Someone should migrate nlab away from Instiki as a wiki format to something that allows multiple kind of edges. 
- Someone should think about linking 
- I would call it the jungle: Generating a lot of AI generated articles in the graph is probably a good idea. I know that humans wil have to sift through it and move stuff from "the jungle" to "the garden" i.e. the verified wiki however for purposes of computation I'm fine with this at first.
- Obisidian is not sufficient either. Sure one can render TikZ as a svg but, it doesn't even show me in what direction the arrow points.
- If one 