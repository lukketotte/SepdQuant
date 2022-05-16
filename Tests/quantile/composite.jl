using Distributions, LinearAlgebra, SpecialFunctions, SepdQuantile, Plots

n = 250;
X = hcat(ones(n), rand(Normal(), n), rand(Normal(), n));
y = X*[0, 1, 1] + rand(Normal(0, 1), n);

par = Sampler(y, X[:,2:3], 0.5, 5000, 5, 1000, 1);
β, θ, σ, α = mcmc(par, 0.5, 0.5, 1., 1, 2, 0.5, [0.,0.]);

β, θ, σ, α = vec(mean(β, dims = 1)), mean(θ), mean(σ), mean(α);

# convert
τ = range(0.1, 0.9, length = 9)
τk = mcτ.(τ, α, θ, σ)

function bkernel(s::Sampler, b::Real, c::AbstractVector{<:Real}, β::AbstractVector{<:Real}, θ::Real, α::Real)
     c.*(s.y-s.X*β.-b) |> z -> (sum((.-z[z.<0]).^θ)/α^θ + sum(z[z.>0].^θ)/(1-α)^θ)
end

function bkernel(s::Sampler, b::Real, β::AbstractVector{<:Real}, θ::Real, α::Real)
     (s.y-s.X*β.-b) |> z -> (sum((.-z[z.<0]).^θ)/α^θ + sum(z[z.>0].^θ)/(1-α)^θ)
end

function logβCond(β::AbstractVector{<:Real}, s::Sampler, C::AbstractMatrix{<:Int}, b::AbstractVector{<:Real},
        θ::Real, σ::Real, α::AbstractVector{<:Real})
        res = 0
        for i in 1:length(b)
            res += bkernel(s, b[i], C[i,:], β, θ, α[i])
        end
    return - gamma(1+1/θ)^θ/σ^θ * res
end

logβCond(β, par, C, b[1,:], θ, σ, τk)
∂β(β, par, C, b[1,:], θ, σ, τk)
∂β2(β, par, C, b[1,:], θ, σ, τk)

∂β(β::AbstractVector{<:Real}, s::Sampler, θ::Real, σ::Real) = ForwardDiff.gradient(β -> logβCond(β, s, θ, σ), β)

function ∂β(β::AbstractVector{<:Real}, s::Sampler, C::AbstractMatrix{<:Int}, b::AbstractVector{<:Real},
        θ::Real, σ::Real, α::AbstractVector{<:Real})
    ForwardDiff.gradient(β -> logβCond(β, s, C, b, θ, σ, α), β)
end

∂β2(β::AbstractVector{<:Real}, s::Sampler, θ::Real, σ::Real) = ForwardDiff.jacobian(β -> -∂β(β, s, θ, σ), β)

function ∂β2(β::AbstractVector{<:Real}, s::Sampler, C::AbstractMatrix{<:Int}, b::AbstractVector{<:Real},
        θ::Real, σ::Real, α::AbstractVector{<:Real})
    ForwardDiff.jacobian(β -> -∂β(β, s, C, b, θ, σ, α), β)
end


function sampleβ(β::AbstractVector{<:Real}, ε::Real,  s::Sampler, C::AbstractMatrix{<:Int}, b::AbstractVector{<:Real},
        θ::Real, σ::Real, α::AbstractVector{<:Real})
    ∇ = ∂β(β, s, C, b, θ, σ, α)
    #H = real((∂β2(β, s, maximum([θ, 1.01]), σ))^(-1) |> Symmetric)
    H = Symmetric((∂β2(β, s, C, b, θ, σ, α)))^(-1)
    prop = β + ε^2 * H / 2 * ∇ + ε * H^(0.5) * vec(rand(MvNormal(zeros(length(β)), I), 1))
    ∇ₚ = ∂β(prop, s, C, b, θ, σ, α)
    #Hₚ = real((∂β2(prop, s, maximum([θ, 1.01]), σ))^(-1) |> Symmetric)
    Hₚ = Symmetric(∂β2(prop, s, C, b, θ, σ, α))^(-1)
    αᵦ = logβCond(prop, s, C, b, θ, σ, α) - logβCond(β, s, C, b, θ, σ, α)
    αᵦ += - logpdf(MvNormal(β + ε^2 / 2 * H * ∇, ε^2 * H), prop)
    αᵦ += logpdf(MvNormal(prop + ε^2/2 * Hₚ * ∇ₚ, ε^2 * Hₚ), β)
    return αᵦ > log(rand(Uniform(0,1), 1)[1]) ? prop : β
end

function logbcond(b::Real, s::Sampler, c::AbstractVector{<:Real}, β::AbstractVector{<:Real}, θ::Real, σ::Real, α::Real)
    return -(gamma(1+1/θ)/σ)^θ * bkernel(s, b, c, β, θ, α)
end

function logbcond(b::V, s::Sampler, C::AbstractMatrix{<:Int}, β::V, θ::Real, σ::Real, α::V) where {V <: AbstractVector{<:Real}}
    res = 0
    for i in 1:length(b)
        res += -(gamma(1+1/θ)/σ)^θ * bkernel(s, b[i], c, β, θ, α[i])
    end
    return res
end

sum(C, dims = 2)

logbcond(b[1,:], par, C[5,:], β, θ, σ, τk)

logbcond(-0.25, par, rand(Binomial(), n), β, σ, θ, τk[3])

function sampleb(b::Real, ε::Real, s::Sampler, c::AbstractVector{<:Real}, β::AbstractVector{<:Real},
        θ::Real, σ::Real, α::Real)
    prop = rand(Normal(b, ε^2))
    logbcond(prop, s, c, β, θ, σ, α) - logbcond(b, s, c, β, θ, σ, α) >= log(rand(Uniform())) ? prop : b
end

sampleb(0.9, 0.1, par, rand(Binomial(), n), β, σ, θ, τk[3])

w = rand(Dirichlet([0.1 for i in 1:K]))
C = rand(Multinomial(1, w), n)

w + vec(sum(C, dims = 2))

function sampleW(C::AbstractMatrix{<:Real}, α::AbstractVector{<:Real})
    n = vec(sum(C, dims = 2))
    rand(Dirichlet(α + n))
end

sampleW(C, ones(length(τk)))

b = range(-0.8, 0.8, length=length(p))
b = rand(Normal(), K)
w = rand(Dirichlet([0.1 for i in 1:K]))
probs = zeros(length(τk))
for i ∈ 1:length(τk)
    probs[i] = w[i]*exp(-(gamma(1+1/θ)/σ)^θ *bkernel(par, b[i], β, θ, τk[i]))
end
probs
probs = probs ./ sum(probs)

rand(Multinomial(1, probs), n)




function sampleC(s::Sampler, b::V, β::V, w::V, α::V, θ::Real, σ::Real) where {V <: AbstractVector{<:Real}}
    probs = zeros(length(α))
    for i ∈ 1:length(α)
        probs[i] = w[i]*exp(-(gamma(1+1/θ)/σ)^θ * bkernel(par, b[i], β, θ, α[i]))
    end
    rand(Multinomial(1, probs./sum(probs)), n)
end

sampleC(par, rand(Uniform(), 9), β, w, τk, θ, σ)

#sampleβ([0.2, 1], 0.2, par, C, b[1,:], θ, σ, τk)

# test
N = 1000
K = length(τk)
b = zeros(N,K)
β = zeros(N, size(par.X, 2))
b[1,:] = rand(Normal(), K)
C = zeros(Int64, K, n, N)
w = zeros(N, K)
w[1,:] = rand(Dirichlet([0.1 for i in 1:K]))
C[:,:,1] = sampleC(par, b[1,:], β[1,:], w[1,:], τk, θ, σ)


for i ∈ 2:N
    β[i,:] = sampleβ(β[i-1,:], 0.2, par, C[:,:,i-1], b[i-1,:], θ, σ, τk)
    for j ∈ 1:K
        b[i,j] = sampleb(b[i-1,j], 0.1, par, C[j,:,i-1], β[i,:], σ, θ, τk[j])
    end
    w[i,:] = sampleW(C[:,:,i-1], [0.1 for i in 1:K])
    C[:,:,i] = sampleC(par, b[i,:], β[i,:], w[i,:], τk, θ, σ)
end

plot(β[:,2])

vec(mean(w, dims = 1))
vec(mean(C, dims = [2,3]))

plot(b[:,findmax(vec(mean(C, dims = [2,3])))[2]])

p = plot(b[:,1]);
for k in 2:K
    p = plot!(b[:,k])
end
p
