using Distributions, LinearAlgebra, StatsBase, SpecialFunctions, ForwardDiff
using Plots, PlotThemes, CSV, DataFrames, StatFiles, CSVFiles
theme(:juno)
using Turing, QuantileRegressions
include("../../QuantileReg/QuantileReg.jl")
include("../aepd.jl")
using .QuantileReg, .AEPD


n = 200;
β, α, σ = [2.1, 0.8], 0.5, 2.;
θ =  1.
X = [repeat([1], n) rand(Uniform(10, 20), n)]
y = X * β .+ rand(Aepd(0., σ, θ, 0.5), n);

##
function δ(α::Real, θ::Real)
    return 2*(α*(1-α))^θ / (α^θ + (1-α)^θ)
end

function logβCond(β::AbstractVector{<:Real}, s::Sampler, θ::Real, σ::Real, τ::Real, λ::AbstractVector{<:Real})
    z = s.y - s.X*β
    b = δ(s.α, θ)/σ * (sum((.-z[z.< 0]).^θ) / s.α^θ + sum(z[z.>=0].^θ) / (1-s.α)^θ)
    return -b -1/(2*τ) * β'*diagm(λ.^(-2))*β
end

# first derivative
∂β(β::AbstractVector{<:Real}, s::Sampler, θ::Real, σ::Real, τ::Real, λ::AbstractVector{<:Real}) = ForwardDiff.gradient(β -> logβCond(β, s, θ, σ, τ, λ), β)
∂β2(β::AbstractVector{<:Real}, s::Sampler, θ::Real, σ::Real, τ::Real, λ::AbstractVector{<:Real}) = ForwardDiff.jacobian(β -> -∂β(β, s, θ, σ, τ, λ), β)

##
diagm([1; [10 for i in 1:5]])^(-1)

function hmc(β::AbstractArray{Float64}, θ::Real, σ::Real, τ::Real, s::Sampler; L = 40, ϵ = 0.001, p = size(s.X, 2))
    λ = abs.(rand(Cauchy(0,1), p))
    #M = diagm([1; [10 for i in 1:(p-1)]])^(-1) |> Symmetric
    M = diagm([.3, 1, .5, .7, 1000, 1, .5, 1000, .1])^(-1)
    ϕ = reshape(rand(MvNormal(zeros(p), M), 1), p) + 0.5 * ϵ * ∂β(β, s, θ, σ, τ, λ)
    β₁, ϕ₁ = β, ϕ
    log_p = logβCond(β₁, s, θ, σ, τ, λ) - ϕ₁'*M*ϕ₁/2 # sum(ϕ₁.^2)/2
    for l ∈ 1:L
        ϕ += 0.5 * ϵ * ∂β(β, s, θ, σ, τ, λ)
        β += ϵ * M * ϕ
        ϕ += 0.5 * ϵ * ∂β(β, s, θ, σ, τ, λ)
    end
    log_p_L = logβCond(β, s, θ, σ, τ, λ) - ϕ'*M*ϕ/2 # sum(ϕ.^2)/2
    r = exp(log_p_L - log_p)
    r > rand(Uniform()) ? β : β₁
end

par = Sampler(y, X, 0.5, 1000, 1, 1);
inits = coef(qreg(@formula(y ~ x2 + x3 + x4 + x5 + x6 + x7 + x8 + x9), hcat(DataFrame(y = log.(y)), DataFrame(X, :auto)), 0.5))

N = 4001
b = zeros(N, size(X,2))
b[1,:] = inits
for i ∈ 2:N
    b[i,:] = hmc(b[i-1,:], 1.8, 4.5, 100, par; L = 40, ϵ = 0.001)
end
p = 8
plot(β[:,p])
plot!(b[:,p])

p = 1
plot(1:N, cumsum(b[:,p])./(1:N))
plot!(1:N, cumsum(β[:,p])./(1:N))

1-((b[2:N, 2] .=== b[1:(N - 1), 2]) |> mean)
chain = Chains(b, ["intercept";names(dat[:, Not(["osvAll"])])]);
mean(summarystats(chain)[:, :ess]) / N
[median(b[:,i]) for i in 1:size(X,2)]
println(inits)
