using BenchmarkTools
using LenticulumCore
using LinearAlgebra: I
using Random: Xoshiro

const SUITE::BenchmarkGroup = BenchmarkGroup()
core = SUITE["LenticulumCore"] = BenchmarkGroup()
beliefs = core["gaussian"] = BenchmarkGroup()
energies = core["energy"] = BenchmarkGroup()

function gaussian_fixture()
    rng = Xoshiro(41)
    W = randn(rng, 8, 8) / 4
    return Gaussian(randn(rng, 8), Matrix(I + W' * W))
end

beliefs["mean"] = @benchmarkable belief_mean(b) setup=(b = gaussian_fixture())
beliefs["logpartition"] = @benchmarkable logpartition(b) setup=(b = gaussian_fixture())
beliefs["marginalise"] = @benchmarkable marginalise(b, ix) setup=((b, ix) = (extended(gaussian_fixture(), 8), [1, 3, 5, 7]))
energies["squared_norm"] = @benchmarkable scalarise(s, e) setup=((s, e) = (SquaredNorm(), randn(Xoshiro(8), 16)))
