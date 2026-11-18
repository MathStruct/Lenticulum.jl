# GH-22: whether a factor has an `m_step` is asked with the concrete arguments of the call site, so a
# method specialised on the parameter type (a typed NamedTuple, say) is seen, where asking `which`
# with `Any` for the parameter and freeze arguments hid it. Oracles: dispatch itself.
using Test
using MyceliumLearning
using MyceliumLearning: MomentStats, m_step, _has_m_step

struct TypedParamFactor end
struct PlainFactor end
struct OtherParamsFactor end

MyceliumLearning.m_step(::TypedParamFactor, ::MomentStats, ps::NamedTuple{(:a,)}, frozen) = ps
MyceliumLearning.m_step(::OtherParamsFactor, ::MomentStats, ps::NamedTuple{(:b,)}, frozen) = ps

@testset "GH-22: m_step detection uses the concrete call-site arguments" begin
    st = MomentStats([:x], [1], 1)
    @test m_step(TypedParamFactor(), st, (a = 1,), ()) == (a = 1,)
    @test _has_m_step(TypedParamFactor(), (a = 1,), ())                 # specialised on the parameter type
    @test !_has_m_step(TypedParamFactor(), (b = 1,), ())                # ...but not for other parameters
    @test !_has_m_step(PlainFactor(), (a = 1,), ())                      # only the fallback: no M-step
    @test _has_m_step(OtherParamsFactor(), (b = 2.0,), (:A, :x))        # any freeze list, concrete
    @test m_step(PlainFactor(), st, (a = 1,), ()) === nothing            # the fallback returns nothing
end
