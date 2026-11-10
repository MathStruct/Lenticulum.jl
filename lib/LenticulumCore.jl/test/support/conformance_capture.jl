# Capture intentional checker failures without recording them in the package suite.
# The caller asserts their number, so a checker that silently skips a section fails the test.
struct ConformanceCapture <: Test.AbstractTestSet
    description::String
    results::Vector{Any}
end
ConformanceCapture(description; kwargs...) = ConformanceCapture(description, Any[])
Test.record(ts::ConformanceCapture, result) = push!(ts.results, result)
function Test.finish(ts::ConformanceCapture)
    parent = Test.get_testset()
    parent isa ConformanceCapture && Test.record(parent, ts)
    return ts
end

function capture_conformance_checks(check)
    return @testset ConformanceCapture "intentional conformance rejection" begin
        check()
    end
end

conformance_failures(ts::ConformanceCapture) = sum(ts.results; init=0) do result
    result isa Test.Error && error("unexpected error in captured checker: $result")
    result isa ConformanceCapture ? conformance_failures(result) : result isa Test.Fail
end
