# The relay enforces x = y. The marginal on either end is the product of
# the two priors, and every replica can also be run as its own graph.
function coverage_plate_template!(b, i; detach = false, tie = true)
    variable!(b, :x); variable!(b, :y)
    if b isa PlateBuilder
        factor!(b, :px, PriorFactor(:x, LenticulumCore.Gaussian(i, 2), _ -> 0.0); tie)
    else
        factor!(b, :px, PriorFactor(:x, LenticulumCore.Gaussian(i, 2), _ -> 0.0))
    end
    factor!(b, :py, PriorFactor(:y, LenticulumCore.Gaussian(3, 1), _ -> 0.0))
    factor!(b, :relay, RelayFactor(:x, :y))
    connect!(b, :px, :x, :x; direction = Emitting())
    connect!(b, :py, :y, :y; direction = Emitting())
    connect!(b, :relay, :x, :x)
    connect!(b, :relay, :y, :y; detach)
end

# Compare actual edge messages, including missing entries and their timestamps.
function check_plate_sweep!(s, ref, g, schedule; damping = 0.0, ps = (;), st = (;))
    r, state = sweep!(s, g, schedule, ps, st; damping)
    rref, stateref = sweep!(ref, g, SequentialSchedule(tasks(schedule)), ps, st; damping)
    # The untied stateless path preserves absent entries. An absent subtree
    # and an explicitly empty one both supply the same state to its factor.
    @test all(k -> get(state, k, (;)) == get(stateref, k, (;)), parameter_keys(g))
    @test r ≈ rref
    for (actual, expected) in ((s.to_variable, ref.to_variable), (s.to_factor, ref.to_factor))
        for (a, b) in zip(actual, expected)
            if b === nothing
                @test a === nothing
            else
                @test a.iteration == b.iteration
                @test belief_distance(a.belief, b.belief) ≈ 0 atol = 1e-10
                if b.belief isa LenticulumCore.GaussianBelief
                    @test a.belief.improper == b.belief.improper
                end
            end
        end
    end
    return state
end

@testset "Plate replicas and grouped schedule fallbacks" begin
    nt = (;)
    for n in (1, 3)
        b = GraphBuilder()
        pl = plate!(coverage_plate_template!, b, :rep, n)
        g = build(b)
        @test size(pl.edges) == (4, n)
        @test size(pl.factors) == (3, n)
        @test size(pl.variables) == (2, n)
        sch = plate_schedule(g)
        @test Set(tasks(sch)) == Set(all_tasks(g))
        s, ref = MessageStore(g), MessageStore(g)
        check_plate_sweep!(s, ref, g, sch)
        check_plate_sweep!(s, ref, g, sch)
        for i in 1:n
            plain = GraphBuilder()
            coverage_plate_template!(plain, i)
            gp = build(plain)
            m, report, _, _ = infer!(gp, tree_schedule(gp), nt, nt)
            @test report.converged
            for (j, name) in enumerate((:x, :y))
                bel = marginal(s, g, pl.variables[j, i])
                @test belief_distance(bel, getproperty(m, name)) ≈ 0 atol = 1e-12
                @test only(LenticulumCore.belief_mean(bel)) ≈ (i + 6) / 3
                @test only(LenticulumCore.belief_cov(bel)) ≈ 2 / 3
            end
        end
        if n > 1
            @test all(!isnothing, s.to_variable.blocks)
            @test Mycelium._row_block(s.to_factor, 4) isa GaussianBlock
        end
    end

    b = GraphBuilder()
    pl = plate!(b, :rep, 3) do p, i
        coverage_plate_template!(p, i; tie = false)
    end
    g = build(b)
    sch = plate_schedule(g)
    # Same rows allocated separately, each direction reversed separately, and no rows.
    rows = [collect(pl.edges[k, :]) for k in axes(pl.edges, 1)]
    edge_messages(rs) = Mycelium.EdgeMessages(nedges(g), rs)
    for (tv, tf) in ((rows, copy(rows)), (reverse(rows), rows), (rows, reverse(rows)),
                     (reverse(rows), reverse(rows)))
        s, ref = MessageStore(edge_messages(tv), edge_messages(tf), 0), MessageStore(g)
        for _ in 1:2
            check_plate_sweep!(s, ref, g, sch; damping = 0.25)
        end
        @test Mycelium._row_block(s.to_factor, findfirst(==(rows[4]), tf)) isa GaussianBlock
        @test (s.to_factor.blocks[findfirst(==(rows[4]), tf)] !== nothing) == (tv == tf == rows)
    end
    for ntree in (nt, NamedTuple{(replica_name(:px, 2),)}(((extra = 1,),)))
        source = MessageTask(pl.edges[1, 1], :to_variable)
        s = MessageStore(Any[nothing for _ in 1:nedges(g)], Any[nothing for _ in 1:nedges(g)], 0)
        ref = MessageStore(g)
        check_plate_sweep!(s, ref, g, sch; ps = ntree)
        check_plate_sweep!(s, ref, g, sch; st = ntree)
        @test isempty(s.to_variable.rows)
        s, ref = MessageStore(g), MessageStore(g)
        lifted = plate_schedule(g, SequentialSchedule([source]))
        check_plate_sweep!(s, ref, g, lifted; ps = ntree)
        @test (s.to_variable.blocks[1] !== nothing) == isempty(ntree)
    end

    # A grouped cavity with a partial row, or with one pinned variable, is computed per edge.
    cavity = plate_schedule(g, [MessageTask(pl.edges[3, 1], :to_factor)])
    for pinned in (false, true)
        s, ref = MessageStore(g), MessageStore(g)
        for store in (s, ref)
            store.to_variable[pl.edges[1, 1]] = Message(LenticulumCore.Gaussian(5.0, 1.0), 0)
            if pinned
                for e in pl.edges[1, 2:end]
                    store.to_variable[e] = Message(LenticulumCore.Gaussian(-2.0, 1.0), 0)
                end
                pin!(store, g, replica_name(:x, 2), [-7.0])
            end
        end
        check_plate_sweep!(s, ref, g, cavity)
        @test s.to_factor.blocks[3] === nothing
        middle = s.to_factor[pl.edges[3, 2]].belief
        if pinned
            @test middle.value == [-7.0]
        else
            @test LenticulumCore.isuninformative(middle)
        end
    end

    # An input may be absent, or may be present but uninformative. Both break
    # uniform availability within a group and require per-replica polarity.
    emit = plate_schedule(g, [MessageTask(pl.edges[4, 1], :to_variable)])
    for (uniform, absent) in ((false, false), (false, true), (true, false))
        s, ref = MessageStore(g), MessageStore(g)
        for store in (s, ref), i in 1:3
            belief = i == 1 && !uniform ?
                     (absent ? nothing : LenticulumCore.uninformative(1)) : LenticulumCore.Gaussian(i, 1)
            belief === nothing || (store.to_factor[pl.edges[3, i]] = Message(belief, 0))
        end
        if uniform
            check_plate_sweep!(s, ref, g, PlateSchedule(emit.groups); damping = 0.2)
            check_plate_sweep!(s, ref, g, PlateSchedule(emit.groups, emit.plans); damping = 0.2)
        else
            check_plate_sweep!(s, ref, g, emit)
            @test s.to_variable[pl.edges[4, 1]].belief isa TrivialBelief
            @test s.to_variable.blocks[4] === nothing
        end
    end
    # The detached target is committed per edge, even on an undamped sweep.
    b = GraphBuilder()
    plate!((p, i) -> coverage_plate_template!(p, i; detach = true), b, :rep, 2)
    detached = build(b)
    s = MessageStore(detached)
    check_plate_sweep!(s, MessageStore(detached), detached, plate_schedule(detached))
    @test s.to_variable.blocks[4] === nothing
end

@testset "Plate construction and lifting refusals" begin
    @test_throws ArgumentError plate!((p, i) -> nothing, GraphBuilder(), :rep, 0)
    @test_throws ArgumentError plate!(GraphBuilder(), :rep, 1) do p, i
        variable!(p, :x)
        connect!(p, :missing, :x, :x)
    end
    @test_throws ArgumentError plate!(GraphBuilder(), :rep, 2) do p, i
        variable!(p, :x)
        factor!(p, :d, DataFactor(:x, i))
        connect!(p, :d, :x, i == 1 ? :x : replica_name(:x, 1))
    end
    @test_throws ArgumentError plate!(GraphBuilder(), :rep, 2) do p, i
        variable!(p, :x)
        factor!(p, :d, DataFactor(:x, i))
        connect!(p, :d, :x, :x; detach = i == 2)
    end

    b = GraphBuilder()
    variable!(b, :shared)
    pl = plate!(b, :rep, 2) do p, i
        factor!(p, :d, PriorFactor(:x, LenticulumCore.Gaussian(i, 1), _ -> 0.0))
        connect!(p, :d, :x, :shared; direction = Emitting())
    end
    factor!(b, :outer, PriorFactor(:x, LenticulumCore.Gaussian(0, 1), _ -> 0.0))
    e = connect!(b, :outer, :x, :shared; direction = Emitting())
    g = build(b)
    @test all(==(g.variable_index[:shared]), [g.edges[id].variable for id in pl.edges])
    @test_throws ArgumentError plate_schedule(g)
    @test_throws ArgumentError plate_schedule(g, [MessageTask(pl.edges[1, 2], :to_variable)])
    @test_throws ArgumentError as_plate(g, :other, 2)
    schedule = plate_schedule(g, [MessageTask(pl.edges[1, 1], :to_variable), MessageTask(e, :to_variable)])
    @test length.(schedule.groups) == [2, 1]
    s = MessageStore(g)
    check_plate_sweep!(s, MessageStore(g), g, schedule)
    @test only(LenticulumCore.belief_mean(marginal(s, g, 1))) ≈ 1
    @test only(LenticulumCore.belief_cov(marginal(s, g, 1))) ≈ 1/3
    # The unplanned runtime check also rejects batching a shared target.
    check_plate_sweep!(MessageStore(g), MessageStore(g), g, PlateSchedule(schedule.groups))

    b = GraphBuilder()
    plate!(b, :loop, 2) do p, i
        variable!(p, :x); variable!(p, :y)
        for name in (:r, :s)
            factor!(p, name, RelayFactor(:x, :y))
            connect!(p, name, :x, :x); connect!(p, name, :y, :y)
        end
    end
    @test_throws ArgumentError plate_schedule(build(b))
end

@testset "as_plate preserves named replicas and keys" begin
    function named_graph(varids, keys; cross = false, outer = false)
        b = GraphBuilder()
        for i in varids
            variable!(b, replica_name(:x, i))
        end
        for (i, key) in enumerate(keys)
            f = replica_name(:data, i)
            factor!(b, f, DataFactor(:x, [i]); params = key)
            connect!(b, f, :x, replica_name(:x, cross ? 3 - i : i); direction = Emitting())
        end
        if outer
            factor!(b, :outside, PriorFactor(:x, LenticulumCore.Gaussian(1, 1), _ -> 0.0))
            connect!(b, :outside, :x, replica_name(:x, 1); direction = Emitting())
        end
        return build(b)
    end
    keys = [replica_name(:data_key, i) for i in 1:2]
    g = named_graph(1:2, keys; outer = true)
    h = as_plate(g, :rep, 2)
    pl = only(plates(h))
    @test [parameter_key(h.factors[id]) for id in pl.factors[1, :]] == keys
    @test Set(parameter_keys(h)) == Set(parameter_keys(g))
    @test nedges(h) == nedges(g) == 3
    @test g.factors[g.factor_index[:outside]].factor === h.factors[h.factor_index[:outside]].factor
    for i in 1:2
        @test h.variables[pl.variables[1, i]].name === replica_name(:x, i)
        @test h.edges[pl.edges[1, i]].direction isa Emitting
        @test h.edges[pl.edges[1, i]].variable == pl.variables[1, i]
    end
    original, _, _, _ = infer!(g, spanning_tree_schedule(g), (;), (;))
    replated, _, _, _ = infer!(h, spanning_tree_schedule(h), (;), (;))
    for i in 1:2
        name = replica_name(:x, i)
        @test getproperty(replated, name).value == getproperty(original, name).value == [i]
    end
    @test_throws ArgumentError as_plate(named_graph(1:1, Symbol[]), :rep, 2)
    @test_throws ArgumentError as_plate(named_graph(1:2, [:first, :second]), :rep, 2)
    @test_throws ArgumentError as_plate(named_graph(1:2, keys; cross = true), :rep, 2)
end

@testset "EP on blocks matches tilted moments and per-edge updates" begin
    gb(η, Λ, improper = false) = LenticulumCore.GaussianBelief([η], fill(Λ, 1, 1), improper)
    # Both components have precision 2. Wide separation gives negative site
    # precision for the symmetric tilt, even though the tilt itself is proper.
    P, Q = fill(1.0, 1, 1), fill(2.0, 1, 1)
    mix = MixtureBelief([LenticulumCore.GaussianBelief([η], Q) for η in (-6.0, 6.0)], [0.5, 0.5])
    b = GraphBuilder()
    pl = plate!(b, :ep, 2) do p, i
        variable!(p, :x)
        factor!(p, :prior, PriorFactor(:x, LenticulumCore.GaussianBelief([i - 1.0], P), _ -> 0.0))
        factor!(p, :lik, PriorFactor(:x, mix, _ -> 0.0); rule = ExpectationPropagationRule())
        connect!(p, :prior, :x, :x; direction = Emitting())
        connect!(p, :lik, :x, :x; direction = Emitting())
    end
    g = build(b)
    prior_task = MessageTask(pl.edges[1, 1], :to_variable)
    site_task = MessageTask(pl.edges[2, 1], :to_variable)
    sch = plate_schedule(g, [prior_task, site_task])
    s, ref = MessageStore(g), MessageStore(g)
    cav = GaussianBlock(reshape([0.0, 1.0], 1, 2), P, [false, false])
    m = MixtureBlock(reshape([-6.0, 6.0, -6.0, 6.0], 1, 2, 2), Q, fill(log(0.5), 2, 2), false)
    sites = ep_block(m, cav)
    @test sites isa GaussianBlock
    @test sites.Λ isa AbstractVector
    for i in 1:2
        site = replica(sites, i)
        col = ep_message(replica(m, i), replica(cav, i))
        @test belief_distance(site, col) ≈ 0 atol = 1e-12
        @test site.improper == col.improper
        # The likelihood component means are ±3. Their tilted weights
        # are proportional to N(μ_cavity; ±3, 1 + 1/2).
        u = i - 1
        w = exp.(-([(-3 - u)^2, (3 - u)^2]) ./ 3)
        w ./= sum(w)
        means = ([-6, 6] .+ u) ./ 3
        μ = sum(w .* means)
        v = 1/3 + sum(w .* (means .- μ).^2)
        @test only(site.η) ≈ μ / v - u atol = 1e-12
        @test only(site.Λ) ≈ inv(v) - 1 atol = 1e-12
        @test site.improper == (inv(v) <= 1)
    end
    @test replica(sites, 1).improper

    # Distinct precision objects force the column fallback even with proper cavities.
    per_column = GaussianBlock(cav.η, [P, copy(P)], cav.improper)
    @test Mycelium.block_distance(ep_block(m, per_column), sites) ≈ 0 atol = 1e-12
    bad_cavity = GaussianBlock(cav.η, [P, -P], [false, true])
    @test ep_message(replica(m, 2), replica(bad_cavity, 2)) === nothing
    @test ep_block(m, bad_cavity) === nothing
    @test ep_block(cav, bad_cavity) === cav

    check_plate_sweep!(s, ref, g, sch)
    @test s.to_variable.blocks[2].belief isa GaussianBlock
    check_plate_sweep!(s, ref, g, sch)
    @test Mycelium.block_distance(s.to_variable.blocks[2].belief, sites) ≈ 0 atol = 1e-12

    only_sites = plate_schedule(g, [site_task])
    for mode in (:no_rows, :partial, :pinned, :skipped)
        s = mode === :no_rows ? MessageStore(copy(ref.to_variable), copy(ref.to_factor), ref.iteration) : MessageStore(g)
        r = mode === :no_rows ? MessageStore(copy(ref.to_variable), copy(ref.to_factor), ref.iteration) : MessageStore(g)
        for store in (s, r)
            if mode === :partial
                store.to_variable[pl.edges[1, 1]] = Message(replica(cav, 1), 0)
            elseif mode === :pinned
                pin!(store, g, replica_name(:x, 1), [0.0])
            elseif mode === :skipped
                for i in 1:2
                    store.to_variable[pl.edges[1, i]] = Message(replica(bad_cavity, i), 0)
                end
                store.to_variable[pl.edges[2, 2]] = Message(gb(0.5, 1.0), 0)
            end
        end
        check_plate_sweep!(s, r, g, only_sites)
        @test all(isnothing, s.to_variable.blocks)
        if mode === :skipped
            @test s.to_variable[pl.edges[2, 1]].iteration == 1
            @test s.to_variable[pl.edges[2, 2]].iteration == 0
            @test only(s.to_variable[pl.edges[2, 2]].belief.η) == 0.5
        end
    end
end

@testset "EP block with a Float32 cavity and Float64 mixture" begin
    m = MixtureBlock(reshape([-6.0, 6.0], 1, 2, 1), fill(2.0, 1, 1),
                     reshape(log.([0.25, 0.75]), 2, 1), false)
    cav = GaussianBlock(reshape(Float32[1], 1, 1), fill(1f0, 1, 1), [false])
    batched = only(replicas(ep_block(m, cav)))
    single = ep_message(replica(m, 1), replica(cav, 1))
    @test batched.Λ ≈ single.Λ atol = 1e-12 rtol = 0
    # The batched site stored η in the cavity's Float32 array, rounding the
    # Float64 result of projection and division (found by the coverage tests, 2026-11-22).
    @test batched.η ≈ single.η atol = 1e-12 rtol = 0
    @test eltype(batched.η) === Float64
end

@testset "EP plate skips only members with indefinite inputs" begin
    b = GraphBuilder()
    pl = plate!(b, :ep, 2) do p, i
        variable!(p, :x); variable!(p, :y)
        factor!(p, :r, RelayFactor(:x, :y); rule = ExpectationPropagationRule())
        connect!(p, :r, :x, :x; direction = Absorbing())
        connect!(p, :r, :y, :y; direction = Emitting())
    end
    g = build(b)
    sch = plate_schedule(g, [MessageTask(pl.edges[2, 1], :to_variable)])
    for representation in (:shared, :per_replica, :heterogeneous, :point)
        s, ref = MessageStore(g), MessageStore(g)
        negative = LenticulumCore.GaussianBelief([2.0], fill(-1.0, 1, 1), true)
        good = LenticulumCore.Gaussian(9.0, 2.0)
        second_input = representation === :shared ? negative : representation === :per_replica ? good : DiracBelief([9.0])
        first_input = representation === :point ? DiracBelief([-2.0]) : negative
        for store in (s, ref)
            store.to_factor[pl.edges[1, 1]] = Message(first_input, 0)
            store.to_factor[pl.edges[1, 2]] = Message(second_input, 0)
        end
        check_plate_sweep!(s, ref, g, sch)
        if representation === :point
            @test s.to_variable.blocks[2].belief isa PointBlock
            @test s.to_variable[pl.edges[2, 1]].belief.value == [-2.0]
        else
            @test s.to_variable[pl.edges[2, 1]] === nothing
        end
        if representation === :shared
            @test s.to_variable[pl.edges[2, 2]] === nothing
        else
            @test belief_distance(s.to_variable[pl.edges[2, 2]].belief, second_input) == 0
        end
    end
end

@testset "Hybrid plate lifting follows the component's hybrid tasks" begin
    b = GraphBuilder()
    pl = plate!(b, :hybrid, 2) do p, i
        variable!(p, :x); variable!(p, :y)
        factor!(p, :px, PriorFactor(:x, LenticulumCore.Gaussian(i, 4), _ -> 0.0))
        factor!(p, :py, PriorFactor(:y, LenticulumCore.Gaussian(0, 4), _ -> 0.0))
        factor!(p, :mf, RecordingFactor(Any[]); rule = MeanField())
        connect!(p, :px, :x, :x); connect!(p, :py, :y, :y)
        connect!(p, :mf, :a, :x); connect!(p, :mf, :b, :y)
    end
    g = build(b)
    e1 = Set(pl.edges[:, 1])
    first_tasks = filter(t -> t.edge in e1, tasks(hybrid_schedule(g)))
    expected = plate_schedule(g, HybridSchedule(first_tasks))
    sch = plate_schedule(g)
    @test tasks(sch) == tasks(expected)
    @test Mycelium._default_criterion(sch) === :marginals
    @test Mycelium._default_criterion(expected) === :marginals
    s = MessageStore(g)
    check_plate_sweep!(s, MessageStore(g), g, sch)
    for i in 1:2
        # RecordingFactor emits N(1,1), while y's prior is N(0,4).
        @test only(LenticulumCore.belief_mean(marginal(s, g, pl.variables[2, i]))) ≈ 4/5
    end
end

# A small factor with controlled batch output to test the batch protocol.
struct PlateBatchProbe <: AbstractLenticulumFactor
    value::Int
    short::Bool
end
LenticulumCore.channels(::PlateBatchProbe) = (LenticulumCore.Channel(:x, nothing),)
LenticulumCore.supports_polarity(::PlateBatchProbe, p::Polarity) = p[:x] isa Unobserved
LenticulumCore.islearnable(::PlateBatchProbe) = false
function Mycelium.factor_message(f::PlateBatchProbe, ::Symbol, _, inputs, prior, ps, st)
    count = get(st, :count, 0)
    return DiracBelief([f.value + count]), (count = count + 1,)
end
function plate_probe_batch(fs)
    members = first(fs).short ? fs[1:1] : fs
    # An untied batch cannot write one state to several independent keys.
    return [DiracBelief([f.value]) for f in members], (count = length(members),)
end
Mycelium.factor_messages(fs::AbstractVector{PlateBatchProbe}, target, polarity,
                         inputs::NamedTuple, priors::AbstractVector, ps, st) = plate_probe_batch(fs)
Mycelium.factor_messages(fs::AbstractVector{PlateBatchProbe}, target, polarity,
                         inputs::AbstractVector, priors::AbstractVector, ps, st) = plate_probe_batch(fs)

@testset "Plate batch result length and untied state" begin
    for short in (false, true)
        b = GraphBuilder()
        plate!(b, :rep, 2) do p, i
            variable!(p, :x)
            factor!(p, :probe, PlateBatchProbe(i, short); tie = short)
            connect!(p, :probe, :x, :x; direction = Emitting())
        end
        g = build(b)
        sch = plate_schedule(g)
        if short
            # A bad override must be rejected on both block and vector inputs.
            @test_throws ArgumentError sweep!(MessageStore(g), g, sch, (;), (;))
            rowless = MessageStore(Any[nothing, nothing], Any[nothing, nothing], 0)
            @test_throws ArgumentError sweep!(rowless, g, sch, (;), (;))
            @test_throws ArgumentError sweep!(MessageStore(g), g, PlateSchedule(sch.groups), (;), (;))
        else
            for store in (MessageStore(g), MessageStore(Any[nothing, nothing], Any[nothing, nothing], 0))
                # A non-empty batch state makes the group fall back to each key's own update.
                st = check_plate_sweep!(store, MessageStore(g), g, sch)
                for i in 1:2
                    @test getproperty(st, replica_name(:probe, i)) == (count = 1,)
                    @test store.to_variable[i].belief.value == [i]
                end
            end
        end
    end
end

# A reader of the graph's edge messages without the optional row storage,
# as in the replay-store test of Lenticulum.
struct PlateEdgeStore <: Mycelium.AbstractMessageStore
    to_variable::Vector{Any}
    to_factor::Vector{Any}
end

@testset "Plate factor energies with an edge-only store" begin
    b = GraphBuilder()
    plate!(b, :observations, 2) do p, i
        variable!(p, :x)
        factor!(p, :prior, PriorFactor(:x, LenticulumCore.Gaussian(0, 1),
                                      x -> (only(x)^2 + log(2π)) / 2))
        factor!(p, :data, DataFactor(:x, [i]))
        connect!(p, :prior, :x, :x)
        connect!(p, :data, :x, :x; direction = Emitting())
    end
    g = build(b)
    nt = (;)
    s = MessageStore(g)
    sweep!(s, g, plate_schedule(g), nt, nt)
    edge_store = PlateEdgeStore(copy(s.to_variable), copy(s.to_factor))
    for store in (s, edge_store)
        energies, _ = factor_free_energies(store, g, nt, nt)
        for i in 1:2
            @test getproperty(energies.parts, replica_name(:prior, i)) ≈ (i^2 + log(2π)) / 2
            @test iszero(getproperty(energies.parts, replica_name(:data, i)))
        end
    end
end
