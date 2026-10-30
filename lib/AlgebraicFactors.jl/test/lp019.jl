# LP-019 fifth review: branch entropy bounds follow Core's Gaussian convention (the precision
# is its symmetric part), so they agree with Mycelium's entropy and accept what isproper accepts.
@testset "LP-019 entropy bounds use the symmetric part of the precision" begin
    for Λ in ([1.0 0.5001; 0.4998 1.0], [1.0 1.0001; 0.9998 1.0])
        S = (Λ + Λ') / 2
        H = (2 * (1 + log(2π)) - logdet(S)) / 2
        m = LenticulumCore.MixtureBelief([GaussianBelief(zeros(2), Λ)], [1.0])
        b = branch_entropy_bounds(m)
        @test b.lower ≈ H && b.upper ≈ H                 # was 2.98178 vs 2.98168, then DomainError
        @test H ≈ Mycelium.variable_entropy(GaussianBelief(zeros(2), Λ))
    end
end
