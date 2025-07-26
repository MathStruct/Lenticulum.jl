A **pseudo commutative 2-diagram** in a [[2-category]] is a [[2-diagram]] where composition along different paths yields [[1-morphism|1-morphisms]] that are connected by invertible [[2-morphism|2-morphisms]] (rather than being strictly equal).

More precisely, given a 2-diagram in a 2-category, it is pseudo commutative if for any two composable paths of 1-morphisms from object A to object B, there exists an invertible 2-morphism (called a [[pseudo-isomorphism]]) relating the composite morphisms obtained by following these different paths.

<!-- \begin{tikzcd} A \arrow[r, "f"] \arrow[d, "g"'] & B \arrow[d, "h"] \\ C \arrow[r, "k"'] & D \end{tikzcd} with invertible 2-morphism: $\alpha: h \circ f \Rightarrow k \circ g$ -->

This is the categorical analogue of a commutative diagram in ordinary [[category theory]], but relaxed to allow for [[equivalence]] rather than strict equality. The invertible 2-morphisms witnessing the pseudo commutativity are part of the structure and must satisfy coherence conditions when multiple such 2-morphisms are composed.