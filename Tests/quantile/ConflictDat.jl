using Distributed, SharedArrays
@everywhere using Distributions, LinearAlgebra, StatsBase, SpecialFunctions, QuantileRegressions, DataFrames, QuadGK
@everywhere include(joinpath(@__DIR__, "../aepd.jl"))
@everywhere include(joinpath(@__DIR__, "../../QuantileReg/QuantileReg.jl"))
@everywhere using .AEPD, .QuantileReg

using Plots, PlotThemes, CSV, StatFiles, CSVFiles
theme(:juno)

@everywhere f(x, b, p, α, μ, σ) = abs(x-b)^(p-1) * pdf(Aepd(μ, σ, p, α), x)

@everywhere function quantconvert(q, p, α, μ, σ)
    a₁ = quadgk(x -> f(x, q, p, α, μ, σ), -Inf, Inf)[1]
    a₂ = quadgk(x -> f(x, q, p, α, μ, σ), -Inf, q)[1]
    1/((maximum([a₁/a₂, 1.0001]) - 1)^(1/p) + 1)
end

@everywhere function mcτ(τ, α, p, σ, n = 1000, N = 1000)
    res = zeros(N)
    for i in 1:N
        dat = rand(Aepd(0, σ, p, α), n)
        q = DataFrame(hcat(dat), :auto) |> x -> qreg(@formula(x1 ~  1), x, τ) |> coef;
        res[i] = quantconvert(q[1], p, α, 0, σ)
    end
    mean(res)
end

dat = load(string(pwd(), "/Tests/data/hks_jvdr.csv")) |> DataFrame;
y, X = dat[:, :osvAll], dat[:, Not(["osvAll"])] |> Matrix;
X = X[y.>0,:];
y = y[y.>0];
X = hcat([1 for i in 1:length(y)], X);

α = range(0.1, 0.9, length = 9)
p, s, a = 1.9659771517710287, 3.925316251127944, 0.37734587781936146
B = [-1.733882520540273, -0.02497382239874957, -2.2174786785070655,
    2.222952091888118, 4.279598568635178e-5, 0.42894516548263745,
    1.2657078863640019, -0.0015088400285910744, 0.1733927024309777]

settings = SharedArray(hcat(α, zeros(9), zeros(9)))

#par = Sampler(y, X, a, 20000, 1, 10000);
#β, θ, σ, α = mcmc(par, 0.8, .2, 1., 1, 2, 0.5, zeros(size(X,2)));

p,s,a,B = median(θ), median(σ), median(α), median(β, dims=1);
μ = par.X * B' |> x -> reshape(x, size(x, 1));
par.nMCMC = 60000
par.burnIn = 20000

@sync @distributed for i ∈ 1:length(α)
    println(α[i])
    b = DataFrame(hcat(par.y, par.X), :auto) |> x ->
        qreg(@formula(x1 ~  x3 + x4 + x5 + x6 + x7 + x8 + x9 + x10), x, α[i]) |> coef
    q = par.X * b
    #par.α = mcτ(α[i], a, p, s, 5000)
    par.α = [quantconvert(q[j], p, a, μ[j], s) for j in 1:length(par.y)] |> mean
    ϵ = α[i] == 0.1  ? 0.7 : 0.85
    β = mcmc(par, ϵ, p, s, zeros(size(par.X, 2)), verbose = false)
    settings[i, 2] = [par.y[i] <= q[i] for i in 1:length(par.y)] |> mean
    settings[i, 3] = (median(β, dims = 1) |> x -> [par.y[i] <= par.X[i,:] ⋅ x  for i in 1:length(par.y)]) |> mean
end

settings

plt_dat = DataFrame(Tables.table(settings)) |> x -> rename!(x, ["quantile", "quantreg", "sepdreg"])
#CSV.write("C:/Users/lukar818/Dropbox/PhD/research/applied/quantile/R/plots/quantcompconflict.csv", plt_dat)
