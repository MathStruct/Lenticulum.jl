# [Theory vault](@id theory)

Everything these pages leave out — the mathematics, the papers, the design decisions, and an
honest record of what does not work yet — lives in the project's **Obsidian vault**. It is
rendered as a website and deployed beside this one:

```@raw html
<blockquote><p><strong><a href="../vault/">Open the theory vault →</a></strong></p></blockquote>
```

## What is in it

The vault is a map of content with about a hundred notes. A few entry points:

| start at | for |
|---|---|
| *Index* | the whole map, in reading order |
| *Lux as a Parametric Lens* | why a factor is not a layer, in terms of code you already use |
| *The Linear Gaussian Chain* | the [getting-started](@ref getting-started) example, worked through properly |
| *Related Julia Projects* | where this sits next to Turing, RxInfer, ModelingToolkit and Catlab — and when to use those instead |
| *Motivating Examples* | six problem domains with the same shape |

Each package here also has **implementation notes** sitting beside its source — `messages.md`
beside `messages.jl`, and so on — recording the difficulties each file ran into and what it
does not do. Those are in the vault too, under *Implementation*.

## Two kinds of documentation, on purpose

These pages describe **the code**: what each package does and how to call it. The vault
describes **why**, and it is deliberately not summarised here — a docstring is the wrong place
for a derivation, and a derivation is the wrong place for a signature.

When a docstring on this site cites something in double brackets, like `[[Bethe Free Energy]]`,
that is a link into the vault.

## Reading it in Obsidian

The vault is the repository itself. Clone it, open the root folder in
[Obsidian](https://obsidian.md), and start from `Start here.md`. Two community plugins are
assumed: *Inline TikZ* for the diagrams (the website compiles these to SVG itself) and
*Wypst*.
