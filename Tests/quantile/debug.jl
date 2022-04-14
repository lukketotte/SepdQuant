using Distributions, LinearAlgebra, StatsBase, SpecialFunctions, ForwardDiff, QuantileRegressions, QuadGK
include("../../QuantileReg/QuantileReg.jl")
include("../aepd.jl")
include("../../QuantileReg/FreqQuantileReg.jl")
using .AEPD, .QuantileReg, .FreqQuantileReg

using Plots, PlotThemes, CSV, DataFrames, StatFiles, CSVFiles, HTTP



using PDMats


kernel(s::Sampler, β::AbstractVector{<:Real}, θ::Real) = s.y-s.X*β |> z -> (sum((.-z[z.<0]).^θ)/s.α^θ + sum(z[z.>0].^θ)/(1-s.α)^θ)

function logβCond(β::AbstractVector{<:Real}, s::Sampler, θ::Real, σ::Real)
    return - gamma(1+1/θ)^θ/σ^θ * kernel(s, β, θ)
end

∂β(β::AbstractVector{<:Real}, s::Sampler, θ::Real, σ::Real) = ForwardDiff.gradient(β -> logβCond(β, s, θ, σ), β)
∂β2(β::AbstractVector{<:Real}, s::Sampler, θ::Real, σ::Real) = ForwardDiff.jacobian(β -> -∂β(β, s, θ, σ), β)

function sampleβ(β::AbstractVector{<:Real}, ε::Real,  s::Sampler, θ::Real, σ::Real)
    ∇ = ∂β(β, s, θ, σ)
    #H = real((∂β2(β, s, maximum([θ, 1.01]), σ))^(-1) |> Symmetric)
    H = try
            (PDMat(Symmetric((∂β2(β, s, maximum([θ, 1.01]), σ)))))^(-1)
        catch e
            if isa(e, PosDefException)
                #A = Symmetric((∂β2(β, s, maximum([θ, 1.01]), σ)))
                #(PDMat(A + I*eigmax(A)))^(-1)
                println("Warning: PosDefException for H")
                (PDMat((s.X's.X) * sum((s.y-s.X*vec(β)).^2)))^(-1)
            end
        end
    prop = β + ε^2 * H / 2 * ∇ + ε * H^(0.5) * vec(rand(MvNormal(zeros(length(β)), I), 1))
    ∇ₚ = ∂β(prop, s, θ, σ)
    #Hₚ = real((∂β2(prop, s, maximum([θ, 1.01]), σ))^(-1) |> Symmetric)
    Hₚ = try
            (PDMat(Symmetric(∂β2(prop, s, maximum([θ, 1.01]), σ))))^(-1)
        catch e
            if isa(e, PosDefException)
                #A = Symmetric((∂β2(β, s, maximum([θ, 1.01]), σ)))
                #(PDMat(A + I*eigmax(A)))^(-1)
                println(prop)
                (PDMat((s.X's.X) * sum((s.y-s.X*vec(prop)).^2)))^(-1)
            end
        end
    αᵦ = logβCond(prop, s, θ, σ) - logβCond(β, s, θ, σ)
    αᵦ += - logpdf(MvNormal(β + ε^2 / 2 * H * ∇, ε^2 * H), prop)
    αᵦ += logpdf(MvNormal(prop + ε^2/2 * Hₚ * ∇ₚ, ε^2 * Hₚ), β)
    return αᵦ > log(rand(Uniform(0,1), 1)[1]) ? prop : β
end

#β, p, σ = mcmc(par, 0.5, 0.5, 1.5, 2, [0., 0., 0.])

## Find quantiles
# τ = 0.1
Gumbel(0, 5)
quantile(Gumbel(4.1701670167016704, 5), 0.1)
loc = range(4, 4.5, length = 10000)

abs.(quantile.(Gumbel.(loc, 5), 0.1)) |> argmin
loc[3404]

loc = range(1, 1.5, length = 10000)
abs.(quantile.(NoncentralT.(6, loc), 0.1)) |> argmin
loc[5631]
histogram(rand(NoncentralT(6, loc[5631]), 10000))
quantile(NoncentralT(6, 1.2815281528152815), 0.1)

## Testing package
# using SepdQuantile, Random, Distributions
ϵ = zeros(n)
val = 1
while val > 0.01
    global ϵ = bivmix(n,  0.88089, -2.5, 1, 0.5, 1)
    global val = abs(sort(ϵ)[Integer(length(ϵ)*0.9)])
end

sort(ϵ)[Integer(length(ϵ)*0.9)]

ϵ = bivmix(n,  0.88089, -2.5, 1, 0.5, 1)
histogram(ϵ)

n = 250;
X = hcat(ones(n), rand(Normal(), n), rand(Normal(), n))
#y = X*ones(3) + rand(Aepd(0, 1, 0.2, 0.5), n)
y = X*ones(3) + rand(Erlang(7, 0.5), n)
par = Sampler(y, X, 0.999, 5000, 5, 1000, 0.5);
β, p, σ = mcmc(par, 0.5, 0.5, 1.5, 2, [0., 0., 0.]);


sampleβ([10.187663381800625, 1.1112038968410358, 1.3741099524097926], 0.5,  par, 0.3487289794689, 7.871871280810961)


0.0002*PDMat((par.X'par.X) * sum((par.y-par.X*vec([8.33561339195197, 1.0725327153283963, 0.7318042484171956])).^2))^(-1)

MvNormal([0,0,0], 0.0002*PDMat((par.X'par.X) * sum((par.y-par.X*vec([8.33561339195197, 1.0725327153283963, 0.7318042484171956])).^2))^(-1))
β = mean(β, dims = 1)
sum((y - X*[β[i] for i in 1:length(β)]).^2)

PDMat((X'X) * sum((y - X*vec(β)).^2))^(-1)

vec(β)
plot(β[:, 1])

y = X*[2, 1, 1] + rand(Normal(-1.281456, 1), n) # shifted so Q_ϵ(0.9)≈0
y = X*ones(3) + (rand(TDist(3), n) .-1.6369)#(1 .+ X[:,2]).*rand(Normal(), n)
y = X*ones(3) + [rand(Uniform()) < 0.8 ? rand(Normal()) : rand(Normal(0, 3)) for i in 1:n]

ϵ = bivmix(n,  0.88089, -2.5, 1, 0.5, 1)
y = X*ones(3) + ϵ#bivmix(n, 0.88, -2.2, 1, 0.5, 1)

par = Sampler(y, X, 0.5, 5000, 5, 1000);
β, θ, σ, α = mcmc(par, 0.5, 0.5, 1., 1, 2, 0.5, [0., 0., 0.]);
par.α = mcτ(0.9, mean(α), mean(θ), mean(σ))


bqr = DataFrame(hcat(par.y, par.X), :auto) |> x -> qreg(@formula(x1 ~  x3 + x4), x, 0.9) |> coef
βlp = mcmc(par, .25, mean(θ), mean(σ), [0., 0., 0.])
blp = mean(βlp, dims = 1)'

mean((blp-ones(3)).^2)
mean((bqr-ones(3)).^2)

mean(par.y .<= par.X * blp)
mean(par.y .<= par.X * bqr)

plot(βlp[:,2])
plot(α)
plot(θ)
plot(β[:,1])

acceptance(βlp)
acceptance(α)
acceptance(β)
acceptance(θ)

##
f(x, b, p, α, μ, σ) = abs(x-b)^(p-1) * pdf(Aepd(μ, σ, p, α), x)

function quantconvert(q, p, α, μ, σ)
    a₁ = quadgk(x -> f(x, q, p, α, μ, σ), -Inf, Inf)[1]
    a₂ = quadgk(x -> f(x, q, p, α, μ, σ), -Inf, q)[1]
    1/((maximum([a₁/a₂, 1.0001]) - 1)^(1/p) + 1)
end

function mcτ(τ, α, p, σ, n = 1000, N = 1000)
    res = zeros(N)
    for i in 1:N
        dat = rand(Aepd(0, σ, p, α), n)
        q = DataFrame(hcat(dat), :auto) |> x -> qreg(@formula(x1 ~  1), x, τ) |> coef;
        res[i] = quantconvert(q[1], p, α, 0, σ)
    end
    mean(res)
end

function mcτ(τ, α, p, σ, n = 1000, N = 1000)
    res = zeros(N)
    for i in 1:N
        dat = rand(Aepd(0, σ, p, α), n)
        q = DataFrame(hcat(dat), :auto) |> x -> qreg(@formula(x1 ~  1), x, τ) |> coef;
        a₁ = mean(abs.(dat .- q).^(p-1))
        a₂ = mean(abs.(dat .- q).^(p-1) .* (dat .< q))
        res[i] = 1/((maximum([a₁/a₂, 1.0001]) - 1)^(1/p) + 1)
    end
    mean(res)
end

histogram(bivmix(10000,  0.5, -1.5, 1.5, 1, 1))

dat = HTTP.get("https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-max-temperatures.csv") |> x -> CSV.File(x.body) |> DataFrame
scatter(dat[1:(size(dat,1)-1), :Temperature], dat[2:size(dat,1), :Temperature])

y = log.(dat[2:size(dat, 1),:Temperature])
X = hcat(ones(length(y)), log.(dat[1:(size(dat,1)-1),:Temperature]))

par = Sampler(y, X, 0.5, 10000, 1, 1000);
β, θ, σ, α = mcmc(par, .3, 0.11, 1.1, 2, 1, 0.5, rand(size(par.X, 2)));

mcτ(0.7, mean(α), mean(θ), mean(σ))
mcτ2(0.7, mean(α), mean(θ), mean(σ))

par.α = mcτ2(0.7, mean(α), mean(θ), mean(σ), 5000)
#par.nMCMC, par.burnIn = 6000, 1000
βres = mcmc(par, 0.5, mean(θ), mean(σ), rand(size(par.X, 2)))
mean(par.y .<= par.X *  median(βres, dims = 1)')
