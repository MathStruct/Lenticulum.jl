# SLAM and Sensor Fusion

> The canonical factor-graph application, and the one this project already has a worked example
> of ([[The Linear Gaussian Chain]]). It is also where the competition is strongest, so it is
> the fairest test of whether the framework adds anything.

## The relations

A robot's trajectory and its map are related by geometry, and none of those relations has a
preferred direction:

| relation | says |
|---|---|
| odometry | consecutive poses differ by a measured motion |
| landmark observation | a pose and a landmark are related by a bearing/range |
| loop closure | two poses separated in time are the *same place* |
| calibration | sensor frames are related by fixed unknown transforms |

Written as residuals these are exactly ``r_\theta(\cdot) \approx 0``, and the noise model turns
each into a factor. That is the standard formulation — GTSAM, iSAM2, Caesar.jl all do this.

## The name contains both directions

**S**imultaneous **L**ocalisation **A**nd **M**apping: given the map, find the pose; given the
pose, find the map. Two questions, one set of relations, different quantities clamped.

That is [[Channels and Polarity]] in its most literal instance. In a conventional
implementation the two are separate code paths; here they are one factor asked twice.

## What is observed

Wheels, IMU, cameras, lidar, GPS — at different rates, with different latencies, and none of
them synchronised. This is the standard motivation for continuous-time trajectory estimation,
and it is exactly what [[Time as a Base]] describes: a variable carrying a *trajectory* with a
Gauss–Markov prior, queryable at any time by interpolation rather than by inventing a pose
variable per measurement.

## What would be learned

- A **visual odometry front end** — a network producing a relative pose with a covariance.
- A **learned depth or scene prior** — a diffusion model over plausible geometry, entering as a
  `DiffusionFactor`.
- **Data association** as a soft, learned relation rather than a hard assignment.

The point is that these sit in the same graph as the exact geometric constraints, and the
posterior spans both. A learned front end with a badly calibrated covariance currently
poisons a classical SLAM back end silently; here its contribution is a factor with an energy
you can attribute.

## What a residual means

A nonzero residual is a **loop-closure inconsistency** or a **bad data association** — the two
failure modes that matter in practice. Per-factor energies say *which* constraint disagrees,
which is what a robust back end needs in order to reject it.

## What would be hard

- **Loop closures make the graph loopy**, which is the whole point of them. Loopy message
  passing gives exact means and wrong variances ([[Loopy Message Passing]]) — and in SLAM the
  covariance is used for data association, so a wrong covariance causes wrong associations,
  which cause wrong loop closures. The failure compounds.
- **Poses live on ``SE(2)``/``SE(3)``**, not in a vector space. Beliefs would need to be
  Gaussians in a tangent space with a retraction, and `combine` would need to agree about which
  tangent space. `GaussianBelief` is flat.
- **The competition is excellent.** GTSAM and `IncrementalInference.jl` are mature, and the
  latter already handles non-Gaussian multimodal beliefs with clique recycling for incremental
  updates ([[Related Julia Projects]] §6).

The honest niche is therefore not "another SLAM back end" but **SLAM with learned components
and calibrated uncertainty over both halves** — which is the part the existing tools are least
equipped for.

Related: [[Motivating Examples]], [[The Linear Gaussian Chain]], [[Time as a Base]],
[[Channels and Polarity]], [[Loopy Message Passing]], [[Related Julia Projects]]
