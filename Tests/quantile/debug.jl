using Distributions, LinearAlgebra, StatsBase, SpecialFunctions, ForwardDiff

include("../../QuantileReg/QuantileReg.jl")
include("../../aepd.jl")
using .QuantileReg, .AEPD

n = 5000
p, a, s = 1.21, 0.1, 1

p, a, s = 1.9, 0.444, 4.07
p, a, s = median(θ), median(α), median(σ)
dat = rand(Aepd(0, s, p, a), n)

par2 = Sampler(dat, hcat(ones(n)), a, 5000, 4, 1000);
init = DataFrame(hcat(par2.y), :auto) |> x -> qreg(@formula(x1 ~  1), x, par2.α) |> coef;
bet, _ =  mcmc(par2, 1., p, s, init);
acceptance(bet)
plot(bet[:,1])

n = 1000
res = zeros(1000)
for i in 1:1000
    dat = rand(Aepd(0, s, p, a), n)
    q = DataFrame(hcat(dat), :auto) |> x -> qreg(@formula(x1 ~  1), x, 0.9) |> coef;
    res[i] = quantconvert(q[1], p, a, 0, s)
end
τ = mean(res)

dat = rand(Aepd(0, s, p, a), n)
q = DataFrame(hcat(dat), :auto) |> x -> qreg(@formula(x1 ~  1), x, 0.7) |> coef;
τ = quantconvert(q[1], p, a, 0, s)


par2.α = τ

mean(par2.y .< median(bet[:,1]))
mean(par2.y .< q[1])
