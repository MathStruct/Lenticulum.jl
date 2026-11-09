# The resistive divider of `test/runtests.jl` (`divider_graph`), with its resistors as arguments,
# the message store written as one vector (scalar variables), and a schedule's fixed point. Shared by
# the `bp_*.jl` benchmarks.
using Lenticulum, LenticulumCore, Mycelium, LinearAlgebra, Printf
using LenticulumCore: GaussianBelief

const Vs = 10.0
const CircQ = fill(1e-3, 1, 1)
const Ones1 = ones(1, 1)

function divider_graph(R₁, R₂, pv)
    wp = fill(pv, 1, 1); b = GraphBuilder()
    for v in (:v, :i1, :i2); variable!(b, v, 1); end
    factor!(b, :ohm1, LinearConstraintFactor((v = 1, i1 = 1), 1; noise = CircQ))
    factor!(b, :ohm2, LinearConstraintFactor((v = 1, i2 = 1), 1; noise = CircQ))
    factor!(b, :kcl,  LinearConstraintFactor((i1 = 1, i2 = 1), 1; noise = CircQ))
    for (f, a, c) in ((:ohm1, :v, :i1), (:ohm2, :v, :i2), (:kcl, :i1, :i2))
        connect!(b, f, a, a; direction = Bidirectional()); connect!(b, f, c, c; direction = Bidirectional())
    end
    for v in (:v, :i1, :i2)
        factor!(b, Symbol(:reg, v), GaussianPrior(v, [0.0], wp))
        connect!(b, Symbol(:reg, v), v, v; direction = Bidirectional())
    end
    g = validate(build(b))
    ps = merge((ohm1 = (A = (v = Ones1, i1 = fill(R₁, 1, 1)), c = [Vs]),
                ohm2 = (A = (v = Ones1, i2 = fill(-R₂, 1, 1)), c = [0.0]),
                kcl  = (A = (i1 = Ones1, i2 = -Ones1), c = [0.0])),
               NamedTuple(Symbol(:reg, v) => NamedTuple() for v in (:v, :i1, :i2)))
    st = merge((ohm1 = NamedTuple(), ohm2 = NamedTuple(), kcl = NamedTuple()),
               NamedTuple(Symbol(:reg, v) => NamedTuple() for v in (:v, :i1, :i2)))
    return g, ps, st
end

# the store as one vector: (η, Λ) of every message; a message not yet Gaussian reads as NaN
_slots(s) = [(k, e) for k in (:to_factor, :to_variable) for e in eachindex(s.to_factor)]
_msgs(s, k) = k === :to_factor ? s.to_factor : s.to_variable
function getstate(s)
    x = Float64[]
    for (k, e) in _slots(s)
        m = _msgs(s, k)[e]
        b = m === nothing ? nothing : m.belief
        b isa GaussianBelief ? append!(x, (b.η[1], b.Λ[1])) : append!(x, (NaN, NaN))
    end
    return x
end
function setstate!(s, x)
    for (i, (k, e)) in enumerate(_slots(s))
        m = _msgs(s, k)[e]
        _msgs(s, k)[e] = Mycelium.Message(GaussianBelief([x[2i-1]], fill(x[2i], 1, 1), m.belief.improper), m.iteration)
    end
    return s
end

# a schedule's fixed point: its sweeps run until the residual has not improved for 200 sweeps
# (its rounding floor), returned as a store and as a vector
function fixed_store(g, ps, st, sched; cap = 400_000)
    s = MessageStore(g); best, kbest = Inf, 0
    for k in 1:cap
        r, _ = sweep!(s, g, sched, ps, st)
        r < best && ((best, kbest) = (r, k))
        (r == 0 || k - kbest > 200) && break
    end
    return s
end
fixed_point(g, ps, st, sched; kw...) = getstate(fixed_store(g, ps, st, sched; kw...))
