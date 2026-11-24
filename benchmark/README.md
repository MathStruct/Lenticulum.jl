# Pull request benchmarks

The gate runs the head checkout's benchmark definitions against each revision's
library code. Every warm case records the BenchmarkTools trial, including
median time, allocation count and bytes.

| Suite | Workloads |
|---|---|
| ImplicitSolvers | Dense and Krylov linear solves, a ForwardDiff solve, nonlinear roots and column solves |
| LenticulumCore | Gaussian moments, log partition, marginalisation and energy scalarisation |
| Mycelium | A 96-step Kalman chain with tree inference and fixed flooding passes, scalar free energy, a 10 × 10 grid sweep, a factor-analysis plate E-step and a Gaussian belief product |
| MyceliumLearning | Zygote and ForwardDiff chain gradients, dense and GMRES implicit graph adjoints with ForwardDiff, and a regime plate M-step |
| Lenticulum | Gaussian, hard-constraint and nonlinear messages, and two EM iterations on a regime plate |
| VariationalDiffusion | A seeded small relation learner queried singly and in a batch, and a Zygote gradient through the query |

First-call latency includes `using` for each of these packages, graph
inference plus a Zygote gradient on a fresh Kalman chain, a relation query and
its gradient in separate processes, and regime plate EM. Every latency key
uses three fresh Julia processes after environment instantiation. The minimum,
median and individual samples are saved. Workload setup takes place before
the timer, and the timed call includes compilation.

## Run locally

Use Julia 1.12 or later. From the checkout that supplies the benchmark definitions:

```sh
JULIA_NUM_THREADS=1 julia --project=benchmark benchmark/compare.jl --base dev-danielb
```

The default head uses the current checkout's library files. Pass
`--head REV` to use a detached revision instead. `--also-base REV` adds a
second baseline in the same run and reuses the head results.
`--interleave` measures each revision twice in the same order and selects
the faster median per key, taking allocation and byte counts from the
selected trial. BLAS uses one thread. Use `--report FILE.md` to choose the
report location. The raw JSON files go alongside it and preserve the
BenchmarkTools serialization under `trials`, as well as the latency results,
run metadata and suite errors. The runner includes environment setup in the
wall time.

Use `--packages A,B` or `--filter REGEX` to narrow a run. The filter uses
`Package/group/case` for warm cases and `latency/Package/workload` for
latency. `--samples-time SECONDS` controls the warm sampling budget per
case. Run the tooling smoke check with:

```sh
LENTICULUM_SMOKE=true JULIA_NUM_THREADS=1 julia --project=benchmark benchmark/compare.jl --aa HEAD --packages ImplicitSolvers
```

`julia --project=benchmark benchmark/smoke.jl` runs this comparison and
checks its artifacts. Smoke mode uses four warm samples at most and one
fresh process per latency key. The separate study runner still discovers `bench/provenance.jl` as a
no-op. Each study prints provenance when launched as the entry point.

## Thresholds and calibration

A warm-time flag requires the slower median to exceed the faster one by
more than `max(0.10, 3 × noise)` relative to the faster median. An allocation
increase is flagged above 10%, or on any increase for keys matching
`allocations.budgeted` patterns in `thresholds.toml`. A first-call latency
increase is flagged above 20% relative to the base median.

A matched key uses its value in `benchmark/noise.toml` when present.
Otherwise it uses the default noise of 0.05. A/A runs compare the same
revision twice and write the per-key warm noise, `abs(log(t₂ / t₁))`,
to the chosen `--noise` file. The Markdown report includes the median,
maximum and keys above 0.05. To calibrate CI, use **Actions → Benchmarks →
Run workflow** and enable `aa`. The workflow ref supplies the suite definitions.
Set `head` to a different library revision if needed, such as `origin/dev`.
Download the `benchmark-results` artifact, review the full A/A report, and
copy its `noise.toml` to `benchmark/noise.toml`. Commit the reviewed file with
the CI run link in the commit message. Use full-sized CI results for this file.

The PR job compares against the target branch and, where different, master
on the same machine. The report goes to the step summary and is uploaded with
the raw JSON. Flags are advisory. A failed head suite or a tooling error
makes the command exit nonzero. `--strict` also gives a nonzero exit for
flags. Absent base packages and failed base cases are reported. Surviving keys are still compared.

The gate covers CPU workloads. GPU timings, the Q12 allocation budgets,
and Q15's precompile workloads and their cost measurements are outside its scope.
