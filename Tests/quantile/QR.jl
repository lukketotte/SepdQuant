module QR

export sampleLatent, sampleσ, sampleθ, sampleβ, θinterval

using Distributions, LinearAlgebra, StatsBase, SpecialFunctions

function δ(α::T, θ::T)::T where {T <: Real}
    2*(α*(1-α))^θ / (α^θ + (1-α)^θ)
end

function sampleLatent(X::Array{T, 2}, y::Array{T, 1}, β::Array{T, 1}, α::T, θ::T, σ::T) where {T <: Real}
    n,_ = size(X)
    u₁ = zeros(n)
    u₂ = zeros(n)
    for i in 1:n
        μ = X[i,:] ⋅ β
        if y[i] <= μ
            l = ((μ - y[i]) / (σ * α))^θ
            u₁[i] = rand(truncated(Exponential(1/δ(α, θ)), l, Inf), 1)[1]
        else
            l = ((y[i] - μ) / (σ * (1-α)))^θ
            u₂[i] = rand(truncated(Exponential(1/δ(α, θ)), l, Inf), 1)[1]
        end
    end
    u₁, u₂
end

function rtruncGamma(n::N, a::N, b::T, t::T) where {N, T <: Real}
    v, w = zeros(a), zeros(a);
    v[1], w[1] = 1,1;
    for k in 2:a
        v[k] = v[k-1] * (a-k+1)/(t*b)
        w[k] = w[k-1] + v[k]
    end
    wt = v./w[a]
    x = zeros(n)
    for i in 1:n
        u = rand(Uniform(), 1)[1]
        k = any(wt .>= u) ? minimum(findall(wt .>= u)) : a
        x[i] = t * (rand(InverseGamma(k, 1/(t*b)), 1)[1] + 1)
    end
    x
end

# TODO: is this sampled correctly?
function sampleσ(X::Array{T, 2}, y::Array{T, 1}, u₁::Array{T, 1}, u₂::Array{T, 1},
    β::Array{T, 1}, α::T, θ::T, a::N = 1, b::T = 1.) where {T, N <: Real}
    n = length(y)
    lower = zeros(n)
    for i in 1:n
        μ = X[i,:] ⋅ β
        if (u₁[i] > 0) && (y[i] < μ)
            lower[i] = (μ - y[i]) / (α * u₁[i]^(1/θ))
        elseif (u₂[i] > 0) && (y[i] >= μ)
            lower[i] = (y[i] - μ) / ((1-α) * u₂[i]^(1/θ))
        end
    end
    # rand(Pareto(a + n - 1, maximum(lower)), 1)[1]
    rtruncGamma(1, a + n - 1, b, maximum(lower))[1]
end

function θinterval(X::Array{T, 2}, y::Array{T, 1}, u₁::Array{T, 1}, u₂::Array{T, 1},
    β::Array{T, 1}, α::T, σ::T) where {T <: Real}

    id_pos = findall(((X*β .- y) .> 0) .& (u₁ .> 0))
    id_neg = findall(((y.-X*β) .> 0) .& (u₂ .> 0))
    ids1 = id_pos[findall(log.((X[id_pos,:]*β - y[id_pos])./(σ*α)) .< 0)]
    ids2 = id_pos[findall(log.((X[id_pos,:]*β - y[id_pos])./(σ*α)) .> 0)]
    ids3 = id_neg[findall(log.((y[id_neg]-X[id_neg,:]*β)./(σ*(1-α))) .< 0)]
    ids4 = id_neg[findall(log.((y[id_neg]-X[id_neg,:]*β)./(σ*(1-α))) .> 0)]

    l1 = length(ids1) > 0 ? maximum(log.(u₁[ids1])./log.((X[ids1,:]*β - y[ids1])./(α*σ))) : 0
    l2 = length(ids3) > 0 ? maximum(log.(u₂[ids3])./log.(( y[ids3]-X[ids3,:]*β)./((1-α)*σ))) : 0

    up1 = length(ids2) > 0 ? minimum(log.(u₁[ids2]) ./ log.((X[ids2,:]*β - y[ids2])./(α*σ))) : Inf
    up2 = length(ids4) > 0 ? minimum(log.(u₂[ids4]) ./log.((y[ids4]-X[ids4,:]*β)./((1-α)*σ))) : Inf

    [maximum([0 l1 l2]) minimum([up1 up2])]
end


function θcond(θ::T, u₁::Array{T, 1}, u₂::Array{T, 1}, α::T) where {T <: Real}
    n = length(u₁)
    n*(1+1/θ) * log(δ(α, θ))  - n*log(gamma(1+1/θ)) - δ(α, θ) * sum(u₁ .+ u₂)
end

function sampleθ(θ::T, ε::T, X::Array{T, 2}, y::Array{T, 1}, u₁::Array{T, 1}, u₂::Array{T, 1},
    β::Array{T, 1}, α::T, σ::T) where {T <: Real}
    interval = θinterval(X, y, u₁, u₂, β, α, σ)

    """d = truncated(Normal(θ, ε), minimum(interval), maximum(interval))
    prop = rand(d, 1)[1]
    gPrev = logpdf(truncated(Normal(prop, ε),minimum(interval), maximum(interval)), θ)
    gProp = logpdf(d, prop)
    θcond(prop, u₁, u₂, α) - θcond(θ, u₁, u₂, α) + gPrev - gProp >= log(rand(Uniform(0,1), 1)[1]) ? prop : θ"""

    prop = rand(Uniform(minimum(interval), maximum(interval)))
    θcond(prop, u₁, u₂, α) - θcond(θ, u₁, u₂, α) >= log(rand(Uniform(0,1), 1)[1]) ? prop : θ
end

function sampleβ(X::Array{T, 2}, y::Array{T, 1}, u₁::Array{T, 1}, u₂::Array{T, 1},
    β::Array{T, 1}, α::T, θ::T, σ::T, τ::T) where {T <: Real}
    n, p = size(X)
    βsim = zeros(p)
    for k in 1:p
        l, u = [], []
        for i in 1:n
            a = (y[i] - X[i, 1:end .!= k] ⋅  β[1:end .!= k]) / X[i, k]
            b₁ = α*σ*(u₁[i]^(1/θ)) / X[i, k]
            b₂ = (1-α)*σ*(u₂[i]^(1/θ)) / X[i, k]
            if (u₁[i] > 0) && (X[i, k] < 0)
                append!(l, a + b₁)
            elseif (u₂[i] > 0) && (X[i, k] > 0)
                append!(l, a - b₂)
            elseif (u₁[i] > 0) && (X[i, k] > 0)
                append!(u, a + b₁)
            elseif (u₂[i] > 0) && (X[i, k] < 0)
                append!(u, a - b₂)
            end
        end
        length(l) == 0. && append!(l, -Inf)
        length(u) == 0. && append!(u, Inf)
        βsim[k] = maximum(l) < minimum(u) ? rand(truncated(Normal(0, τ), maximum(l), minimum(u)), 1)[1] : maximum(l)
    end
    βsim
end

end
