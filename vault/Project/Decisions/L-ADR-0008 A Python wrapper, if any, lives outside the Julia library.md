#design

> **Status: proposed (2026-10-29, Claude, from a discussion with Daniel; low priority).** Related: [[L-ADR-0001 Julia]], [[Why Julia]], [[L-SPEC-001 The Factor Interface]], [[LP-002 The Jacobian side is not implemented]]. Repository: `MathStruct/Lenticulum.py` (empty).

## Context

Most ML users work in Python, and a Julia library asks them to install a second language just
to try an idea. The empty repository `MathStruct/Lenticulum.py` reserves the name for a Python
package. Daniel set two conditions on any such package:

- it must not constrain the performance or functionality of the Julia library in any way;
- if it exists, it is a first-rate Python library on a current stack (uv, pixi, optionally
  Nix), not a thin, hard-to-install shim.

What would be lost in Python (the assessment from the discussion; none of it measured yet):

- **Extensibility.** New factor and belief types written in Python work only through callbacks
  that cross the language boundary on every message. Message passing makes many small calls,
  so such graphs would be slow. Building graphs from Python is fine; writing new factor types
  belongs in Julia.
- **The type discipline.** Polarities and channels are types in Julia
  ([[The Type Discipline of a Factor Graph]]); in Python they become runtime values, and
  errors arrive as Julia stack traces unless translated.
- **Latency.** The first call compiles. PrecompileTools workloads (which help Julia users too)
  or a prebuilt Julia image reduce it.
- **Forward-mode AD through foreign models.** ForwardDiff duals cannot pass through PyTorch or
  JAX, so factors wrapping them support reverse mode only.

PyTorch and JAX models as factors are feasible because a factor needs a forward pass and a
VJP, and the implicit adjoints of LP-002 need only VJPs. One Julia-side factor type would wrap a
Python callable: forward through DLPack (zero copy, GPU included when both sides share the CUDA
context), backward as a ChainRules rule calling `torch.autograd` or `jax.vjp`, which Zygote
then uses. The boundary costs microseconds per call: acceptable for large batched networks,
bad for many tiny factors. JVPs and HVPs through foreign models would be later work. No
maintained general PyTorch/JAX → Lux bridge is known to us; PyCallChainRules.jl did this over
PyCall and its maintenance status is unchecked.

Prior art: PySR (Python front end, SymbolicRegression.jl back end, via PythonCall.jl/juliacall)
is the pattern to follow.

An open question, raised by Daniel: with much code now written by AI agents, it is unclear how
much a Python front end matters. An agent can write Julia for a user as readily as Python, which
weakens the language-barrier argument; but users' existing pipelines, data loaders and models
are in Python, which an agent cannot wish away. There is no evidence either way yet.

## Options

1. **No Python package.** Julia only; Python users call it through juliacall themselves.
2. **A separate Python package** in `Lenticulum.py`, depending on the Julia packages through
   juliacall/juliapkg; the Julia library knows nothing about it (at most an optional
   `PythonCall` package extension for conversions).
3. **Python bindings inside this repository** or Julia-side design choices made for Python's
   sake.

## Decision

Proposed: option 2, and **not yet**. Option 3 is ruled out by the first condition. Start only
when both hold:

- the factor interface (L-SPEC-001) and the inference API have stopped changing, so the
  wrapper does not double the cost of every interface change;
- there is a demand signal: issues or discussions asking for Python, or a concrete user.

If started, the stack is: `pyproject.toml` packaged with uv; Julia dependencies pinned in
`juliapkg.json`; pixi for full environments including Julia and GPU builds of torch/JAX
(optionally a Nix flake for CI); ruff, a strict type checker (pyright or ty), pytest,
mkdocs-material; numpy and DLPack at the boundary; wheels on PyPI and conda-forge. The Python API
takes and returns Python objects (arrays, dataclasses, `torch.nn.Module`), not raw Julia
handles, and its test suite repeats the Julia suite's closed-form checks through the wrapper.

Not accepted until Daniel accepts it.

## Consequences

- Nothing changes in the Julia library now. PrecompileTools workloads are worth adding anyway
  and would serve a wrapper later.
- When the conditions are met, the first deliverables are graph construction and inference
  from Python, then the foreign-model factor (forward + VJP), each measured for boundary
  overhead before any performance claim is made.
