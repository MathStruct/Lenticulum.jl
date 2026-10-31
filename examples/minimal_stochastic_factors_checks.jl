# Numerical checks behind the corrections of vault/Design/Minimal Stochastic Factors.md (§4, §6, §13):
# the inversion of a conditional-Gaussian kernel against a prior other than its reference marginal,
# and the independence of the Born marginal from the POVM. Run: julia examples/minimal_stochastic_factors_checks.jl
using LinearAlgebra, Printf
N(x, m, v) = exp(-(x - m)^2 / (2v)) / sqrt(2π * v)
# --- 1. Prop 3: inversion of the CG kernel c(y|x) against a prior π(x) ≠ P_X ------------------
# reference joint: two components, each Gaussian in (x, y) with y = g_k x + h_k + noise
β = [0.4, 0.6]; μ = [-1.0, 2.0]; vx = [0.5, 1.5]; g = [1.0, -0.5]; h = [0.0, 1.0]; vy = 0.2
PX(x) = sum(β[k] * N(x, μ[k], vx[k]) for k in 1:2)
c(y, x) = sum(β[k] * N(x, μ[k], vx[k]) * N(y, g[k] * x + h[k], vy) for k in 1:2) / PX(x)
prior(x) = N(x, 0.5, 4.0)
y0 = 0.7
xs = range(-12, 12; length = 24001); dx = step(xs)
true_post = [prior(x) * c(y0, x) for x in xs]; true_post ./= sum(true_post) * dx
# my old claim: "π(x) β_k N_k(x, y) per pair", i.e. dropping 1/P_X(x)
old = [prior(x) * sum(β[k] * N(x, μ[k], vx[k]) * N(y0, g[k] * x + h[k], vy) for k in 1:2) for x in xs]; old ./= sum(old) * dx
@printf("1. L1 distance between the true inversion and the claimed closed form: %.4f\n", sum(abs.(true_post .- old)) * dx)
# with π = P_X the inversion is the CG conditional P(x | y): check
post_PX = [PX(x) * c(y0, x) for x in xs]; post_PX ./= sum(post_PX) * dx
cg = [sum(β[k] * N(x, μ[k], vx[k]) * N(y0, g[k] * x + h[k], vy) for k in 1:2) for x in xs]; cg ./= sum(cg) * dx
@printf("   with π = P_X: L1 distance to the CG conditional: %.2e\n", sum(abs.(post_PX .- cg)) * dx)
# --- 2. Born joint: the marginal p(z) does not depend on the POVM ------------------------------
z̃(z) = [1.0, z]
pz(z, E) = N(z, 0.3, 1.0) * sum(dot(z̃(z), Ek * z̃(z)) for Ek in E)
v = normalize([0.4, 1.0]); E1 = [v * v', I - v * v']
w = normalize([1.0, -2.0]); E2 = [w * w', I - w * w']
@printf("2. max |p(z; POVM 1) − p(z; POVM 2)| on a grid: %.2e\n", maximum(abs(pz(z, E1) - pz(z, E2)) for z in -5:0.01:5))
# ... but weighted outcomes (a factor on k) tilt by z̃ᵀ(Σ wₖEₖ)z̃, and M = vvᵀ gives a zero
pw(z) = N(z, 0.3, 1.0) * dot(z̃(z), (1.0 * E1[1] + 0.0 * E1[2]) * z̃(z))
zero_at = -v[1] / v[2]
@printf("   with weights (1, 0): density at the predicted zero z = %.3f is %.2e\n", zero_at, pw(zero_at))
