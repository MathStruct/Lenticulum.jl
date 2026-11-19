# Noise-free nodes against nodes with noise

Run `julia -t auto --project=bench lib/VariationalDiffusion.jl/examples/nodes_calibration.jl` from the
repository root (about 35 minutes on 8 threads, nearly all of it in the queries with noisy nodes;
`LENTICULUM_SMOKE=true` checks the script at small size). This is backlog row P10: compare
`noisefree_nodes` with `field_nodes` on the calibration benchmark ([[Backlog F Families]] F2,
`calibration.md`) and choose a default for the nodes of a root query.

## Question

`field_nodes` averages the noise predictor over noisy inputs $\alpha_t z + \sigma_t\varepsilon$, that is, the
potential $-\log p_t$ over shifts; to second order that adds a Laplacian term of the size of the
smoothing itself, so on a thin ring whose width is small against $\sigma_t/\alpha_t$ the radial bias is
about twice that of a noise-free node ([[Implicit Diffusion Learners]] §5: the ratio is
$(s^2+\sigma_t^2/\alpha_t^2)/(s^2+2\sigma_t^2/\alpha_t^2)$, which tends to 1 at levels small against the width). Noise-free nodes evaluate
$\varepsilon_\theta(\alpha_t z, t)$ once per level. Do they recover the relations of the benchmark better, and
what do they cost?

## Setup

The relations, models (same seeds), queries and metrics of `calibration.jl`, shared through
`calibration_setup.jl`: the unit circle, the lemniscate of Gerono, two circles (0.9 / 0.1); $x$
clamped on a grid, 33 starts per query, $\tau = 0.1$. Without retraining, each model is queried with

- **noise-free**: `noisefree_nodes(2; levels)`, 8 nodes (one per level);
- **with noise**: `field_nodes(Xoshiro(1), 2; samples = 8, levels)`, 128 nodes (8 antithetic pairs
  per level, unbatched: 128 network evaluations per field), as in `calibration.jl`; `field_nodes`' own
  default is `samples = 4` (64 nodes), so the times compare against the calibration configuration, not
  against that default; at the default levels also a second draw, `Xoshiro(2)`;

over three level ranges of 8 levels each: low ($t\le0.01$), default ($t\le0.05$) and mid ($t\le0.1$).
Reported for all roots and for the roots the ridge residual keeps (codimension 1; of the filters of F3
the one that needs no held-out data, and the one that kept the lemniscate's recall at the low levels).
Time is the wall time of one query grid; the noise-free grid ran first for each relation and level range,
so it may carry compilation the noisy grid after it does not (the first grid of the circle, 11 s, against
3 s at the next range).

## Measured (2026-11-19)

All roots; *spurious (ridge)* is the spurious fraction after the ridge filter, which never changed
recall by more than 0.1 and removed every off-domain root in every row (before it, every row has 8 of
8 off-domain queries answered). *with noise* gives the two draws at the default levels.

| relation | levels | nodes | recall | spurious | spurious (ridge) | dist | radial | time |
|---|---|---|---|---|---|---|---|---|
| circle | low | noise-free | 0.857 | 0.234 | 0.077 | 0.0197 | −0.0088 | 11 s |
| circle | low | with noise | 0.857 | 0.234 | 0.077 | 0.0207 | −0.0099 | 60 s |
| circle | default | noise-free | **0.929** | 0.191 | 0.026 | 0.0209 | **−0.0085** | 3 s |
| circle | default | with noise | 0.833 / 0.905 | 0.255 / 0.213 | 0.103 / 0.051 | 0.0285 / 0.0290 | −0.0196 / −0.0184 | 36 / 36 s |
| circle | mid | noise-free | 0.881 | 0.213 | 0.051 | 0.0265 | −0.0174 | 2 s |
| circle | mid | with noise | 0.714 | 0.356 | 0.194 | 0.0693 | −0.0582 | 45 s |
| lemniscate | low | noise-free | **0.571** | 0.586 | 0.400 | 0.0567 | | 6 s |
| lemniscate | low | with noise | 0.524 | 0.621 | 0.421 | 0.0543 | | 87 s |
| lemniscate | default | noise-free | 0.381 | 0.724 | 0.595 | 0.0444 | | 4 s |
| lemniscate | default | with noise | 0.286 / 0.286 | 0.786 / 0.789 | 0.694 / 0.694 | 0.0593 / 0.0588 | | 98 / 84 s |
| lemniscate | mid | noise-free | 0.190 | 0.837 | 0.724 | 0.0446 | | 6 s |
| lemniscate | mid | with noise | 0.214 | 0.828 | 0.786 | 0.0614 | | 24 s |
| two circles | low | noise-free | 0.282 | 0.739 | 0.435 | 0.0164 | +0.0029 | 52 s |
| two circles | low | with noise | 0.282 | 0.739 | 0.435 | 0.0151 | +0.0034 | 451 s |
| two circles | default | noise-free | 0.290 | 0.702 | 0.163 | 0.0250 | −0.0135 | 18 s |
| two circles | default | with noise | 0.282 / 0.282 | 0.703 / 0.706 | 0.146 / 0.167 | 0.0143 / 0.0161 | −0.0104 / −0.0109 | 266 / 301 s |
| two circles | mid | noise-free | 0.298 | 0.479 | 0.143 | 0.0338 | −0.0272 | 36 s |
| two circles | mid | with noise | 0.250 | 0.530 | 0.225 | 0.0572 | −0.0504 | 279 s |

The minority circle (radius 2) is recalled in no row but one (noise-free, mid, all roots: 0.012, one
matched branch entry of the radius-2 circle). The numbers of the first draw of noisy nodes at the low
and default levels repeat those of `calibration.md` exactly (same models, same nodes); the second draw
is new.

## What it shows

1. **Noise-free nodes are 4 to 25 times faster** (from the printed seconds; median 12 over the first
   draws; 8 network evaluations per field instead of 128). The ratio is not 16 everywhere: the solver's
   own work stays, and compilation may fall on the noise-free grid, which ran first.
2. **On the circle they have less bias, more so as the levels grow**: radial −0.0088 against −0.0099 at
   the low levels (89 %), −0.0085 against −0.019 at the default ones (43–46 %), −0.017 against −0.058 at
   mid (30 %). The approach to 1 at the low levels is what §5's formula gives when the levels are small
   against the data's width. Noise-free at the default levels is as precise as noisy nodes at the low
   levels, and has the best recall of the table (0.93).
3. **On the lemniscate they recall more** at the low and default levels (0.57 and 0.38 against 0.52 and
   0.29), with fewer spurious roots; at mid they recall less before the ridge filter (0.19 against 0.21)
   and more after it (0.19 against 0.12); both are poor there.
4. **On the two circles the bias is larger at the default levels**: −0.0135 against −0.0104 (dist 0.025
   against 0.014), where §5 predicts less; recall 0.290 against 0.282. At mid the bias is about half
   (−0.027 against −0.050). Why the default levels differ is not known; a learned model is not the exact
   score that §5's argument assumes.
5. **The field's randomness matters at the size of the differences.** Two draws of the noisy nodes at the
   default levels give circle recall 0.83 and 0.90; noise-free nodes have no such draw.
6. **Off-domain queries are unchanged.** Every off-domain query returns a root with either kind of node,
   and the ridge filter removes all of them with either. The minority circle stays lost with both (the
   one matched entry above is the only exception, and the ridge filter removes it).

## The default

For root queries (`implicit_roots`, `implicit_infer`, `ImplicitProx`), use `noisefree_nodes(n)` at the
default levels ($t\le0.05$). Its recall is as high or higher than the noisy nodes' in eight of the nine
relation and level settings (all roots, first draw), it is deterministic, and it is about 12 times
faster against the calibration configuration; at the default levels its bias is lower on the circle and
higher on the two circles. The noisy nodes remain for what needs an averaged field (a Hutchinson
estimate of the Laplacian, §5) and for `ProxDM`'s free-energy score nodes, whose default in
`DiffusionFactor` stays `field_nodes`: that role was not measured here.

The constructors take their nodes explicitly, so no code default changed; the recommendation is in the
docstrings of `noisefree_nodes` and `field_nodes` and in `src/implicit.md`. The tutorials and the README
still use `field_nodes`, and their recorded outputs were made with it.

## Not checked

One training seed per relation, two draws of noisy nodes at one level range only (noise-free nodes have
no draw), `field_nodes`' own default of 4 pairs per level, one $\tau$, 8 levels per range. Not measured: the free energy and
gradients through `ProxDM` with noise-free score nodes, conditioning of the field and the solver's iteration counts (not
recorded), compilation apart from query time, and relations in more than two dimensions.
