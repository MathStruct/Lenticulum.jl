Please implement the paper [AutoBayes](https://arxiv.org/html/2503.18608v2) as a Julia machine learning library. 
Please start with going through the relevant concepts in the paper and in [Categetorical foundations of Gradient Based learning](https://arxiv.org/html/2103.01931v2#S2)
Also [[ImplicitREDDiff]] contains my previous idea.
Look also at the [[README]] of this whole repository/obsidian vault.

The rough structure of this library is:
- lib/LenticulumCore should mirror LuxCore.jl
- lib/Mycelium.jl should implement the factor graph machinery later needed.
- lib/VariationalDiffusion should contain a LenticulumFactor that uses a Diffusion model in a prox operator to achieve an implicit learner.
- src contains the actual Lenticulum.jl which should mirror Lux.jl


I currently know three category of models that can be implicit learners:
- Based on Algebraic Varieties with help from Differential Algebra for low dimensions
- Equilibrium/Looping Models (DEQ, NeuralODE) (but they need convergencene guarantees)
- Implicit Diffusion Model for High dimensional problems.

You may write the formulas either in Latex or Typst. Can can create TikZ diagrams.
```tikz
\usepackage{tikz-cd}
\begin{document}
\begin{tikzcd}
X \arrow[r, "f"] \arrow[dr, "f"'] \arrow[d, "\mathrm{id}_X"'] & Y \arrow[d, "\mathrm{id}_Y"] \\
X \arrow[r, "f"'] & Y
\end{tikzcd}
\end{document}
```
An addon will automatically render the code.

Please motivate and relate the classical parametric lens based learning to concepts from Lux.jl
please relate the Autobayes concepts to Lenticulum.jl
Please keep two versions of the Energy: The scalar version which is used to calculate losses and a multivariate version which preserves additional structure. I know that the chain rule needs to be adapted but please do so.
Please provide an obsidian vault with all the concepts from the paper. You may write a basic structure of abstract types in LenticulumCore.jl. You can place the markdown files directly next to the Julia implemenation as this is both an obisidan vaulat and a Julia repository. 
Then await instruction for proceeding further.