Why another machine learning framework in Julia?


## Category theory inspired

 Julia's multiple Dispatch enables functionality that


## MLIR compiled

Unlike Haskell Julia is JIT compiled. Which gives it a performance edge and makes it run near C-like speed. 
The newest kid on the block for compiler technologies and automatic differentiation seems to be MLIR. Which is why I will try to build that using this framework. This allows for super fast code as well as parallelization for CPU, GPU, TPU.

## Integration with CatLab.jl



## Testing against BenchmarkTools.jl
I want this model to be competitive which is why I will test it against 
## Taming the Zoo


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
Learners of this kind outperform explicit learners in terms of parameter count and exactness. They can encode and conform to algebraic, differential and geometric constraints in a way explicit learners can't. However they are slower to compute and harder to handle.

## Type Safety

For long term I'm really interested to integrate [Lean.jl](https://github.com/miguelraz/Lean.jl) or similar. This is for two reasons:
- Rigorous type constraints ensure the model stays a diagram and that pieces fit together.
- The models can be subject to formal verification.
## My motivation:

This is my way of learning category theory:
I found the book "The Dao of Functional Programming" by Bartosz Milesvwski and figured that it is a really good approach, because:
**If I learn the concepts in isolation I will forget them.** 
I can sit myself down study what a Monad is. It might take take me a few minutes to figure it out and understand it. But the next time somewhere the term "monad" is mentioned i have to look it up again. 
This repository is my exercise in **practising** categorical concepts by implementing them. 