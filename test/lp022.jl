# LP-022: a constraint factor is normalised in its residual, not in its channels. With every
# coefficient learnable its free energy on data is minimised by A = 0 (every point on the relation)
# and, with learnable noise, unbounded below. `output = :y` fixes the coefficient of the data
# channel and makes the factor a normalised conditional density of y. Oracles: closed forms of
# −Σ log f and least squares.
using Test, Lenticulum, LenticulumCore, Mycelium, LinearAlgebra, LuxCore, Random
using ADTypes, DifferentiationInterface, ForwardDiff
using LenticulumCore: DiracBelief

const L22_N = 200
function l22_data()
    rng = Xoshiro(22)
    x = randn(rng, L22_N)
    return x, 2 .* x .+ 0.5 .+ 0.3 .* randn(rng, L22_N)
end
# point i: x_i and y_i observed, one factor tied over the points
function l22_graph(factor, x, y)
    b = GraphBuilder()
    for i in 1:L22_N
        r(s) = replica_name(s, i)
        variable!(b, r(:x), 1); variable!(b, r(:y), 1)
        factor!(b, r(:dx), DataFactor(:x, [x[i]])); connect!(b, r(:dx), :x, r(:x); direction = Emitting())
        factor!(b, r(:dy), DataFactor(:y, [y[i]])); connect!(b, r(:dy), :y, r(:y); direction = Emitting())
        factor!(b, r(:rel), factor; params = :rel); connect!(b, r(:rel), :x, r(:x)); connect!(b, r(:rel), :y, r(:y))
    end
    return build(b)
end
l22_ps(rel) = merge(NamedTuple(Symbol(replica_name(s, i)) => NamedTuple() for i in 1:L22_N for s in (:dx, :dy)), (rel = rel,))
function l22_F(g, ps)
    s = MessageStore(g)
    for _ in 1:2; sweep!(s, g, SequentialSchedule(all_tasks(g)), ps, NamedTuple()); end
    return first(scalar_free_energy(s, g, ps, NamedTuple())), s
end

@testset "LP-022: learnable coefficients on data channels degenerate" begin
    x, y = l22_data()
    Q = 0.09
    f = LinearConstraintFactor((y = 1, x = 1), 1; noise = Q)
    g = l22_graph(f, x, y)
    # least squares: y = a x + c
    X = [x ones(L22_N)]; a, c = X \ y
    r = y .- a .* x .- c
    rel(Ay, Ax, c) = (A = (y = fill(Ay, 1, 1), x = fill(Ax, 1, 1)), c = [c])
    Ffit, s = l22_F(g, l22_ps(rel(1.0, -a, c)))
    @test Ffit ≈ sum(r .^ 2) / (2Q) + L22_N * log(2π * Q) / 2 rtol = 1e-12
    # A = 0, c = 0: every point is on the relation, and the free energy is lower than the fit's
    F0, _ = l22_F(g, l22_ps(rel(0.0, 0.0, 0.0)))
    @test F0 ≈ L22_N * log(2π * Q) / 2 rtol = 1e-12
    @test F0 < Ffit
    # at the least-squares fit the gradient in A_y is Σ r²/Q > 0: descent shrinks the relation
    G = free_energy_gradient(s, g, l22_ps(rel(1.0, -a, c)), NamedTuple(); ad = AutoForwardDiff()).rel
    @test only(G.A.y) ≈ sum(r .^ 2) / Q rtol = 1e-8
    @test abs(only(G.A.x)) < 1e-8 * L22_N && abs(only(G.c)) < 1e-8 * L22_N
    # with learnable noise there is no lower bound: A → cA, Q → c²Q changes F by N m log c
    fr = RegimeFactor((y = 1, x = 1), 1, 1; noise = Q, learn_noise = true)
    gr = l22_graph(fr, x, y)
    prr(scale) = (A = (y = fill(scale, 1, 1, 1), x = fill(-a * scale, 1, 1, 1)), b = fill(c * scale, 1, 1), logw = [0.0],
                  L = fill(log(sqrt(Q) * scale), 1, 1, 1))
    F1, _ = l22_F(gr, l22_ps(prr(1.0)))
    Fh, _ = l22_F(gr, l22_ps(prr(0.5)))
    @test F1 ≈ Ffit rtol = 1e-12
    @test Fh - F1 ≈ L22_N * log(0.5) rtol = 1e-10
end

@testset "LP-022: a declared output channel is a normalised conditional density" begin
    x, y = l22_data()
    Q = 0.09
    f = LinearConstraintFactor((y = 1, x = 1), 1; noise = Q, output = :y)
    p0 = LuxCore.initialparameters(Xoshiro(1), f)
    @test keys(p0.A) == (:x,) && LuxCore.parameterlength(f) == 2
    g = l22_graph(f, x, y)
    X = [x ones(L22_N)]; a, c = X \ y
    # F is −Σ log N(y; a x + c, Q), the residual being y − a x − c (A_y = 1 fixed)
    rel(Ax, c) = (A = (x = fill(Ax, 1, 1),), c = [c])
    F, s = l22_F(g, l22_ps(rel(-a, c)))
    @test F ≈ sum((y .- a .* x .- c) .^ 2) / (2Q) + L22_N * log(2π * Q) / 2 rtol = 1e-12
    # gradient descent from A_x = 0, c = 0 reaches least squares
    θ = [0.0, 0.0]
    for _ in 1:200
        ps = l22_ps(rel(θ...))
        _, s = l22_F(g, ps)
        G = free_energy_gradient(s, g, ps, NamedTuple(); ad = AutoForwardDiff()).rel
        θ .-= (Q / L22_N) .* [only(G.A.x), only(G.c)]
    end
    @test θ ≈ [-a, c] atol = 1e-8
    # passing the fixed coefficient as a parameter is refused
    @test_throws ArgumentError l22_F(g, l22_ps((A = (y = ones(1, 1), x = fill(-a, 1, 1)), c = [c])))
    # with a fixed non-identity coefficient, the factor is the density up to the constant |det A_y|
    f2 = LinearConstraintFactor((y = 1, x = 1), 1; noise = Q, output = :y, output_coeff = fill(2.0, 1, 1))
    F2, _ = l22_F(l22_graph(f2, x, y), l22_ps(rel(-2a, 2c)))
    @test F2 ≈ sum((2 .* (y .- a .* x .- c)) .^ 2) / (2Q) + L22_N * log(2π * Q) / 2 rtol = 1e-12
end

@testset "LP-022: output channels in the switching and regime factors and in clusters" begin
    # a switching factor with output y and a latent x ~ N(0, 1), in one cluster: y = a x + b_k + ε
    b = GraphBuilder()
    variable!(b, :x, 1); variable!(b, :y, 1); variable!(b, :k, 2)
    factor!(b, :dy, DataFactor(:y, [0.7])); connect!(b, :dy, :y, :y; direction = Emitting())
    factor!(b, :pr, GaussianPrior(:x, 0.0, 1.0)); connect!(b, :pr, :x, :x)
    factor!(b, :sw, SwitchingConstraintFactor((y = 1, x = 1), 2, 1; noise = 0.2, output = :y))
    connect!(b, :sw, :y, :y); connect!(b, :sw, :x, :x); connect!(b, :sw, :k, :k)
    g = build(b)
    psw = (A = (x = fill(-1.5, 1, 1),), b = [-1.0 2.0], logw = log.([0.3, 0.7]))
    @test keys(LuxCore.initialparameters(Xoshiro(1), g.factors[g.factor_index[:sw]].factor).A) == (:x,)
    gc = cluster(g, :c, [:pr, :sw])
    ps = cluster_parameters(gc, :c, (dy = NamedTuple(), pr = NamedTuple(), sw = psw))
    _, _, _, s = infer!(gc, tree_schedule(gc), ps, NamedTuple())
    lp = [log(w) - (0.7 - bk)^2 / (2 * (1.5^2 + 0.2)) - log(2π * (1.5^2 + 0.2)) / 2 for (w, bk) in zip([0.3, 0.7], [-1.0, 2.0])]
    m = maximum(lp)
    @test first(scalar_free_energy(s, gc, ps, NamedTuple())) ≈ -(m + log(sum(exp.(lp .- m)))) rtol = 1e-12
    # the same factor outside a cluster: its message to k given y and x ~ N(0, 1) is the posterior
    fsw = g.factors[g.factor_index[:sw]].factor
    mk, _ = factor_message(fsw, :k, nothing, (y = DiracBelief([0.7]), x = LenticulumCore.Gaussian([0.0], ones(1, 1))), nothing, psw, NamedTuple())
    @test exp.(mk.logp) ≈ exp.(lp .- (m + log(sum(exp.(lp .- m))))) atol = 1e-12
    # RegimeFactor: the output coefficient is fixed in every regime
    fr = RegimeFactor((y = 1, x = 1), 2, 1; output = :y, learn_noise = true)
    pr = LuxCore.initialparameters(Xoshiro(1), fr)
    @test keys(pr.A) == (:x,) && size(pr.A.x) == (1, 1, 2)
    @test Mycelium.cluster_form(fr, pr).A.y == ones(1, 1, 2)
    @test LuxCore.parameterlength(fr) == 2 * (1 + 1 + 1) + 2
    # refusals
    @test_throws ArgumentError LinearConstraintFactor((y = 1, x = 1), 1; output = :z)
    @test_throws ArgumentError LinearConstraintFactor((y = 2, x = 1), 1; output = :y)
    @test_throws ArgumentError LinearConstraintFactor((y = 1, x = 1), 1; output = :y, output_coeff = zeros(1, 1))
    @test_throws ArgumentError SwitchingConstraintFactor((y = 1,), 2, 1; switch = :k, output = :k)
end

@testset "LP-022: free_energy_gradient warns about unnormalised learned relations on data" begin
    # `_unnormalised_nodes` decides the warning (shown once per session, so it is tested here)
    x, y = l22_data()
    gw = l22_graph(LinearConstraintFactor((y = 1, x = 1), 1; noise = 0.09), x, y)
    @test length(Mycelium._unnormalised_nodes(gw)) == L22_N
    @test isempty(Mycelium._unnormalised_nodes(l22_graph(LinearConstraintFactor((y = 1, x = 1), 1; noise = 0.09, output = :y), x, y)))
    @test learns_unnormalised(RegimeFactor((y = 1, x = 1), 2, 1)) && !learns_unnormalised(RegimeFactor((y = 1, x = 1), 2, 1; output = :y))
    # frozen coefficients, or no data, do not warn
    b = GraphBuilder(); variable!(b, :y, 1); variable!(b, :x, 1)
    factor!(b, :dy, DataFactor(:y, [0.3])); connect!(b, :dy, :y, :y; direction = Emitting())
    factor!(b, :rel, LinearConstraintFactor((y = 1, x = 1), 1); freeze = (:A,)); connect!(b, :rel, :y, :y); connect!(b, :rel, :x, :x)
    @test isempty(Mycelium._unnormalised_nodes(build(b)))
    b = GraphBuilder(); variable!(b, :y, 1); variable!(b, :x, 1)
    factor!(b, :pr, GaussianPrior(:x, 0.0, 1.0)); connect!(b, :pr, :x, :x)
    factor!(b, :rel, LinearConstraintFactor((y = 1, x = 1), 1)); connect!(b, :rel, :y, :y); connect!(b, :rel, :x, :x)
    @test isempty(Mycelium._unnormalised_nodes(build(b)))
    # inside a cluster node: the clustered factor is named
    b = GraphBuilder(); variable!(b, :y, 1); variable!(b, :x, 1)
    factor!(b, :dy, DataFactor(:y, [0.3])); connect!(b, :dy, :y, :y; direction = Emitting())
    factor!(b, :pr, GaussianPrior(:x, 0.0, 1.0)); connect!(b, :pr, :x, :x)
    factor!(b, :rel, LinearConstraintFactor((y = 1, x = 1), 1)); connect!(b, :rel, :y, :y); connect!(b, :rel, :x, :x)
    @test Mycelium._unnormalised_nodes(cluster(build(b), :c, [:pr, :rel])) == [Symbol("c/rel")]
end
