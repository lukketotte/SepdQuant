using Distributions, LinearAlgebra, StatsBase, SpecialFunctions, ForwardDiff
include("../../QuantileReg/QuantileReg.jl")
include("../aepd.jl")
include("../../QuantileReg/FreqQuantileReg.jl")
using .AEPD, .QuantileReg, .FreqQuantileReg

## Testing package
using SepdQuantile, Random, Distributions

n = 2000;
x = rand(Normal(), n);
X = hcat(ones(n), x)
y = 2.1 .+ 0.5 .* x + rand(Aepd(0, 1, 1.5, 0.9), n);

par = Sampler(y, X, 0.5, 5000, 5, 1000);
β, θ, σ, α = mcmc(par, 0.8, 0.25, 1.5, 1, 2, 0.5, [2.1, 0.5]);

##
n = 2000;
x = rand(Normal(), n);
X = hcat(ones(n), x)

y = 2.1 .+ 0.5 .* x + rand(Erlang(7, 0.5), n);
y = 2.1 .+ 0.5 .* x + rand(Gumbel(0, 1), n);
y = 2.1 .+ 0.5 .* x + rand(InverseGaussian(1, 1), n); # skip
y = 2.1 .+ 0.5 .* x + rand(TDist(5), n);
y = 2.1 .+ 0.5 .* x + rand(Chi(3), n);

y = 2.1 .+ 0.5 .* x + rand(Aepd(0, 1, 1.5, 0.9), n);

par = Sampler(y, X, 0.5, 5000, 5, 1000);
β, θ, σ, α = mcmc(par, 0.8, 0.25, 1.5, 1, 2, 0.5, [2.1, 0.5]);

par.α = 0.1
par.πθ = "uniform"
par.πθ = "jeffrey"
β, θ, σ = mcmc(par, 0.6, 0.6, 1.0, 2, [2.1, 0.5]);



acceptance(α)
acceptance(β)
acceptance(θ)
plot(β[:, 1])
plot(θ)
plot(α)

μ = X * median(β, dims = 1)' |> x -> reshape(x, size(x, 1));

quant = 0.9
b =
    DataFrame(hcat(par.y, par.X), :auto) |>
    x -> qreg(@formula(x1 ~ x3), x, quant) |> coef;
q = X * b;
τ =
    [
        quantconvert(q[j], median(θ), median(α), μ[j], median(σ)) for
        j = 1:length(par.y)
    ] |> mean

par.α = τ
#par.α = mcτ(0.1, median(α), median(θ), median(σ))
βres, _ = mcmc(par, 1.3, median(θ), median(σ), b);
[par.y[i] <= X[i, :] ⋅ median(βres, dims = 1) for i = 1:length(par.y)] |> mean

acceptance(βres)
plot(βres[:, 2])

# their way
par.α = quant
βt, _, _ = mcmc(par, 0.6, 1.0, 1.2, 4, b);


[par.y[i] <= X[i, :] ⋅ b for i = 1:length(y)] |> mean
[par.y[i] <= X[i, :] ⋅ median(β, dims = 1) for i = 1:length(par.y)] |> mean

√var(βt[:, 2])
√var(βres[:, 2])



n = 2000
res = zeros(n)
for i = 1:n
    dat = rand(Aepd(0, 1, 2, 0.7), n)
    q =
        DataFrame(hcat(dat), :auto) |>
        x -> qreg(@formula(x1 ~ 1), x, 0.1) |> coef
    res[i] = quantconvert(q[1], p, a, 0, s)
end
τ = mean(res)

## double check (this might be the real way to make a point actually)
n = 2000;
x = rand(Normal(), n);
# 2.5, 1.5
y = 2.1 .+ 0.5 .* x + rand(Aepd(0, 3.0, 2.0, 0.7), n);
X = hcat(ones(n), x)

# Freq estimation
control = Dict(
    :tol => 1e-3,
    :max_iter => 1000,
    :max_upd => 0.3,
    :is_se => false,
    :est_beta => true,
    :est_sigma => true,
    :est_p => true,
    :est_tau => true,
    :log => false,
    :verbose => false,
)

typeof(control) <: Dict{Symbol, Real}

res = quantfreq(y, X, control)


control[:est_sigma], control[:est_tau], control[:est_p] = (false, false, false)


b = DataFrame(hcat(y, X), :auto) |> x -> qreg(@formula(x1 ~ x3), x, 0.2) |> coef;
q = X * b;
μ = X * res[:beta]
τ =
    [
        quantconvert(q[j], res[:p], res[:tau], μ[j], res[:sigma]) for
        j = 1:length(y)
    ] |> mean

res = quantfreq(y, X, control, res[:sigma], res[:p], τ)

mean(y .<= X * res[:beta])


res[:p]
res[:sigma]
res[:tau]
res[:beta]

# Simulation stuff
reps = 10
α = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];
res3 = zeros(length(α))
for i ∈ 1:length(α)
    temp = 0
    for j = 1:reps
        par = Sampler(y, X, α[i], 5000, 4, 1000)
        b =
            DataFrame(hcat(par.y, par.X), :auto) |>
            x -> qreg(@formula(x1 ~ x3), x, α[i]) |> coef
        β, _, _ = mcmc(par, 1.0, 1.0, 1.2, 4, b)
        temp +=
            abs(
                α[i] - (
                    [
                        par.y[i] <= X[i, :] ⋅ median(β, dims = 1) for
                        i = 1:length(par.y)
                    ] |> mean
                ),
            ) / reps
    end
    res3[i] = temp
end

plot(α, res3, label = "p = 2.")
plot!(α, res2, label = "p = 1.5")
plot!(α, res3, label = "p = 0.7")

par = Sampler(y, X, α, 5000, 4, 1000);
b =
    DataFrame(hcat(par.y, par.X), :auto) |>
    x -> qreg(@formula(x1 ~ x3), x, α) |> coef;
β, θ, σ = mcmc(par, 1.0, 1.0, 1.2, 4, b);
[par.y[i] <= X[i, :] ⋅ median(β, dims = 1) for i = 1:length(par.y)] |> mean
acceptance(β)
plot(σ)
plot(θ)
plot(β[:, 2])

β, θ, σ, α = mcmc(par, 0.8, 0.25, 1.5, 1, 2, 0.5, b);
q = X * b;
μ = X * median(β, dims = 1)' |> x -> reshape(x, size(x, 1));
τ =
    [
        quantconvert(q[j], median(θ), median(α), μ[j], median(σ)) for
        j = 1:length(par.y)
    ] |> mean

par = Sampler(y, X, τ, 6000, 5, 1000);
βres, _ = mcmc(par, 1.3, median(θ), median(σ), b);

acceptance(βres)

[par.y[i] <= X[i, :] ⋅ b for i = 1:length(y)] |> mean
[par.y[i] <= X[i, :] ⋅ median(β, dims = 1) for i = 1:length(par.y)] |> mean
[par.y[i] <= X[i, :] ⋅ median(βres, dims = 1) for i = 1:length(par.y)] |> mean
