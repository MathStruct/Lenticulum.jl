A **2-diagram** is a configuration of [[object|objects]], [[1-morphism|1-morphisms]], and [[2-morphism|2-morphisms]] in a [[2-category]] that forms a coherent structure.

Specifically, a 2-diagram consists of:

- A collection of objects
- [[1-morphism|1-morphisms]] between these objects
- [[2-morphism|2-morphisms]] between parallel 1-morphisms

The 2-morphisms in the diagram must respect the composition and identity laws of the 2-category. Unlike ordinary [[diagram|diagrams]] in [[category theory]], 2-diagrams can contain 2-morphisms that relate different composite paths, allowing for more flexible notions of commutativity.

<!-- Example 2-diagram: \begin{tikzcd} A \arrow[r, "f"] \arrow[d, "g"'] & B \arrow[d, "h"] \\ C \arrow[r, "k"'] & D \end{tikzcd} with 2-morphism: $\alpha: h \circ f \Rightarrow k \circ g$ -->

2-diagrams are fundamental in [[higher category theory]] and provide the setting for defining concepts like [[pseudo commutative 2-diagram|pseudo commutativity]], [[lax functor|lax functors]], and [[pseudofunctor|pseudofunctors]].