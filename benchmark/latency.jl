# This entry point intentionally imports no packages before timing `using`.
ARGS[2] == "using" || include(joinpath(@__DIR__, "common.jl"))

function main(args)
    pkg, name, out = args
    seconds = if name == "using"
        expr = :(using $(Symbol(pkg)))
        @elapsed Core.eval(Main, expr)
    else
        suite = load_suite(pkg)
        setup, task, args = if name == "kalman_infer_and_gradient"
            (:kalman_gradient_inputs, :kalman_first_call, (3,))
        elseif name == "relation_query"
            (:relation_inputs, :relation_query, ())
        elseif name == "relation_gradient"
            (:relation_inputs, :relation_gradient, ())
        elseif name == "regime_plate_fit_em"
            (:regime_plate, :fit_regime, ())
        else
            error("Unknown latency workload: ", name)
        end
        inputs = Base.invokelatest(Base.invokelatest(getproperty, suite, setup), args...)
        workload = Base.invokelatest(getproperty, suite, task)
        @elapsed Base.invokelatest(workload, inputs...)
    end
    write(out, string(seconds))
end

main(ARGS)
