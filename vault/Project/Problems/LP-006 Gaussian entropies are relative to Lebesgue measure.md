#open-problem

> **Status: accepted, documented (2026-10-22).** Related: [[Scalar and Multivariate Energy]] §8, [Entropy in a Markov Category (CT-ML wiki)](https://mathstruct.org/CategoryTheory-ML-Wiki/Entropy-in-a-Markov-Category), [[Bethe Free Energy]].

**Problem.** The free energies of the Gaussian factors use differential entropies, e.g. $\tfrac12\log\det(2\pi e \Sigma)$. In Perrone's categorical account the KL entropy of a probability measure is Shannon entropy if it is discrete and **infinite** otherwise (*Markov Categories and Entropy*, Corollary 4.7): copying a continuous variable gives a measure on the diagonal, singular with respect to $p \otimes p$. Differential entropy is finite only because it is taken relative to Lebesgue measure, and it changes under reparametrisation.

**Resolution.** Accepted: free energies of continuous beliefs are relative to a reference measure, Lebesgue in the coordinates of each variable. Consequences to keep in mind:
- free energies are comparable between models over the same coordinates, not across reparametrisations;
- a change of variables inside a factor must add its log-Jacobian;
- mixing discrete and continuous variables in one free energy needs care with the reference measures.
