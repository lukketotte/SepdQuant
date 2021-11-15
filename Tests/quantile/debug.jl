using Distributions, LinearAlgebra, StatsBase, SpecialFunctions, ForwardDiff

include("../../QuantileReg/QuantileReg.jl")
include("../../aepd.jl")
using .QuantileReg, .AEPD

n = 5000
p, a, s = 1.21, 0.1, 1

p, a, s = 1.94, 0.4, 4.07
dat = rand(Aepd(0, s, p, a), n)

par2 = Sampler(dat, hcat(ones(n)), a, 5000, 4, 1000);
init = DataFrame(hcat(par2.y), :auto) |> x -> qreg(@formula(x1 ~  1), x, par2.α) |> coef;
bet, _ =  mcmc(par2, 2., p, s, init);
acceptance(bet)
plot(bet[:,1])

q = DataFrame(hcat(par2.y), :auto) |> x -> qreg(@formula(x1 ~  1), x, 0.8) |> coef;
τ = quantconvert(q[1], p, a, mean(bet, dims = 1)[1,1], s)


par2.α = τ

mean(par2.y .< mean(bet[:,1]))
mean(par2.y .< q[1])
