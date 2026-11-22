using ForwardDiff, Mycelium, Random, Test, Zygote
using LenticulumCore: DiracBelief, GaussianBelief
using Mycelium: ComponentMixtureBlock, GaussianBlock, VectorBlock, belief_block, replica

@testset "LP-044: build message blocks under AD" begin
    rng = Xoshiro(44)
    belief_loss(b::GaussianBelief) = (b.improper ? 2 : 1) * (sum(abs2, b.η) + sum(abs2, b.Λ))
    belief_loss(b::DiracBelief) = sum(abs2, b.value)

    for (d, n) in ((1, 1), (2, 3), (3, 4)), with_improper in (false, true)
        x = randn(rng, d * n + d^2 * n)
        function parameters(x, shared)
            η = reshape(x[1:d*n], d, n)
            precision(i) = reshape(x[d*n+(i-1)*d^2+1:d*n+i*d^2], d, d)
            Λ = shared ? precision(1) : [precision(i) for i in 1:n]
            flags = [with_improper && isodd(i) for i in 1:n]
            return η, Λ, flags
        end
        for shared in (true, false)
            direct(x) = GaussianBlock(parameters(x, shared)...)
            function packed(x)
                η, Λ, flags = parameters(x, shared)
                return belief_block([GaussianBelief(η[:, i], shared ? Λ : Λ[i], flags[i]) for i in 1:n])
            end
            for make_block in (direct, packed)
                @test make_block(x) isa GaussianBlock
                loss = x -> begin
                    block = make_block(x)
                    sum(i -> belief_loss(replica(block, i)), 1:n)
                end
                @test only(Zygote.gradient(loss, x)) ≈ ForwardDiff.gradient(loss, x) rtol = 1e-10
            end
        end

        function component_loss(x)
            η, Λs, flags = parameters(x, false)
            b = ComponentMixtureBlock(reshape(η, d, n, 1), Λs, zeros(n, 1), flags)
            return sum(abs2, b.η) + sum(i -> (b.improper[i] ? 2 : 1) * sum(abs2, b.Λs[i]), 1:n)
        end
        @test only(Zygote.gradient(component_loss, x)) ≈ ForwardDiff.gradient(component_loss, x) rtol = 1e-10

        function heterogeneous(x)
            η, Λ, flags = parameters(x, true)
            return belief_block([i <= n ? GaussianBelief(η[:, i], Λ, flags[i]) : DiracBelief([1.0]) for i in 1:n+1])
        end
        @test heterogeneous(x) isa VectorBlock
        loss = x -> begin
            block = heterogeneous(x)
            sum(i -> belief_loss(replica(block, i)), 1:n+1)
        end
        @test only(Zygote.gradient(loss, x)) ≈ ForwardDiff.gradient(loss, x) rtol = 1e-10
    end
end
