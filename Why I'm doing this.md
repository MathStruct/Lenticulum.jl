Why another machine learning framework in Julia?


## MLIR compiled/ Multiple Dispatch

Unlike Haskell: Julia is JIT compiled. Which gives it a performance edge and makes it run near C-like speed. It furthermore allows performance optimizations that are not purely functional.
Unlike C++, Python, Mojo:  Julia comes by default with Multiple Dispatch.
The newest kid on the block for compiler technologies and automatic differentiation seems to be MLIR. Which is why I will try to build that using this framework. This allows for super fast code as well as parallelization for CPU, GPU, TPU.

## Integration with CatLab.jl

Many things in Categorytheory are already implemented in Category theory. Overall the Julia ecosystem is quite sizeable 

## Testing against BenchmarkTools.jl
I want this model to be competitive which is why I will test it against Lux.jl and Flux.jl and other frameworks. If this thing can end in production then
## Taming the Zoo

Basically I have some ideas when it comes to a theory of everything in machine learning. I wanna know if I can assemble it, harmonizing:
- differentiable programming/optimization
- graph searches/SAT solving/symbolic approaches
- Linking data along (Hyper-)graphs
The easiest way is trying to fit ML concepts one by one into the scheme.
## Implicit learners

Lastly i want to push a boundary with this framework:
Currently learning means that we learn a parameterised function:
$$ f_\theta:X \rightarrow Y $$
however many real world problems (think differential equations) are not modelled as a function directly but require an implicit formulation where 
$$ (x, y)\in \mathcal{R}_\theta(X,Y) $$
a parameterised relation holds instead of a function. This is modelled by learning a function:
$$ r_\theta:X\times Y \rightarrow \mathcal{E} $$
where the input is projected into an error space $\mathcal{E}$ ideally minimising:
$$ r_\theta (x,y) = 0 $$
Learners of this kind outperform explicit learners in terms of parameter count and exactness. They can encode and conform to algebraic, differential and geometric constraints in a way explicit learners can't. However they are slower to compute and harder to handle. A very simple example of such an implicit learner is a control system or reinforcement learning.

## Type Safety

For long term I'm really interested to integrate [Lean.jl](https://github.com/miguelraz/Lean.jl) or similar. This is for two reasons:
- Rigorous type constraints ensure the model stays a diagram and that pieces fit together.
	- Also I can include type checks like "commutativity of multiplication", which is hard to do via traits but easy in Lean. 
- The models can be subject to formal verification.
s## My motivation:


## The real Reason

This is my way of learning category theory:
I found the book "The Dao of Functional Programming" by Bartosz Milesvwski and figured that it is a really good approach, because:
**If I learn the concepts in isolation I will forget them.** 
I can sit myself down study what a Monad is. It might take take me a few minutes to figure it out and understand it. But the next time somewhere the term "monad" is mentioned i have to look it up again. 
This repository is my exercise in **practising** categorical concepts by implementing them.
Also I'm learning alot about Julia, MLIR, HPC and parallel programming along the way.

To be clear: I learned a lot already, but right now that is just my personal experiment in learning something. I'm **unsure** if this repository can actually be pulled off.
### Note about myself:
I'm about to finish my master's degree in Math, the curriculum included:
Hilbertspace theory, Optimization on PDE's, Numeric of PDE's, Optimization on Manifolds, Lie Groups/Lie Algebras, Systems/control theory, Machine Learning, Inverse Problems.
I could continue to focus on this and try to look for a PhD or a job that does this or SLAM or inverse problems, but in my estimation this is chasing diminishing returns and right now an over saturated market (atleast in Germany). Which is why I wanna do something else:
- Over the last years I somehow got interested in papers and books in category theory that concern themselves with data bases, cybernetics, functional programming or machine learning.
- For a short time I had a formal verification students job in Minlog/Scheme, getting me into self studying Lean and Metamath a bit.
All this seems like abstract nonsense until you start seeing unifying structures and beneficial cross pollination across seemingly unrelated disciplines.
Since seeing the papers and seeing some disjoint python libraries popping up around the papers I thought that this project is actually a low hanging fruit.
Also I'm interested into engineering a serious project not climbing the corporate or academic ladder. I don't care that's not interesting nor motivating.
In a way this is also my pitch to either apply as a PhD student (under the condition that I write software not too many papers) or for a grant or to employ me.
 

I know hat the current description scream too much of AI but this has a long way to go and if atleast the direction is valuable then please have mercy. 

I really like and appreciate to get feedback on this.

-- Daniel Boigk