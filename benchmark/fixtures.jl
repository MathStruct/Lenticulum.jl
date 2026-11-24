# BenchmarkTools calls setup before every sample. Keep deterministic inputs and
# schedules, and create fresh mutable execution state separately in each setup.
function cached_fixture(build_fixture)
    cache = Ref{Any}(nothing)
    return () -> begin
        isnothing(cache[]) && (cache[] = build_fixture())
        cache[]
    end
end
