module QR

export sampleLatent, sampleσ, sampleθ, sampleβ, θinterval, sampleμ, mcmc

using Distributions, LinearAlgebra, StatsBase, SpecialFunctions, Formatting

function δ(α::T, θ::T)::T where {T <: Real}
    2*(α*(1-α))^θ / (α^θ + (1-α)^θ)
end

function sampleLatent(X::Array{T, 2}, y::Array{T, 1}, β::Array{T, 1}, α::T, θ::T, σ::T) where {T <: Real}
    n,_ = size(X)
    u₁, u₂ = zeros(n), zeros(n)
    μ = X*β
    for i ∈ 1:n
        if y[i] <= μ[i]
            l = ((μ[i] - y[i]) / (σ * α))^θ
            u₁[i] = rand(truncated(Exponential(1/δ(α, θ)), l, Inf), 1)[1]
        else
            l = ((y[i] - μ[i]) / (σ * (1-α)))^θ
            u₂[i] = rand(truncated(Exponential(1/δ(α, θ)), l, Inf), 1)[1]
        end
    end
    u₁, u₂
end

function sampleLatent(y::Array{T, 1}, μ::T, α::T, θ::T, σ::T) where {T <: Real}
    n = length(y)
    u₁, u₂ = zeros(n), zeros(n)
    for i ∈ 1:n
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

## TODO: using this basically seems to push the scale down to 1
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

function sampleσ(X::Array{T, 2}, y::Array{T, 1}, u₁::Array{T, 1}, u₂::Array{T, 1},
    β::Array{T, 1}, α::T, θ::T, a::N = 1, b::T = 1.) where {T, N <: Real}
    n = length(y)
    lower = zeros(n)
    μ = X * β
    for i ∈ 1:n
        if (u₁[i] > 0) && (y[i] < μ[i])
            lower[i] = (μ[i] - y[i]) / (α * u₁[i]^(1/θ))
        elseif (u₂[i] > 0) && (y[i] >= μ[i])
            # TODO: gives really big values for sigma
            lower[i] = (y[i] - μ[i]) / ((1-α) * u₂[i]^(1/θ))
        end
    end
    rand(Pareto(a + n - 1, maximum(lower)), 1)[1], maximum(lower)
end

function sampleσ(y::Array{T, 1}, u₁::Array{T, 1}, u₂::Array{T, 1},
    μ::T, α::T, θ::T, a::N = 1, b::T = 1.) where {T, N <: Real}
    n = length(y)
    lower = zeros(n)
    for i ∈ 1:n
        if (u₁[i] > 0) && (y[i] < μ)
            lower[i] = (μ - y[i]) / (α * u₁[i]^(1/θ))
        elseif (u₂[i] > 0) && (y[i] >= μ)
            lower[i] = (y[i] - μ) / ((1-α) * u₂[i]^(1/θ))
        end
    end
    # rtruncGamma(1, a + n - 1, b, maximum(lower))[1], maximum(lower)
    rand(Pareto(a + n - 1, maximum(lower)), 1)[1], maximum(lower)
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

function θinterval(y::Array{T, 1}, u₁::Array{T, 1}, u₂::Array{T, 1},
    μ::T, α::T, σ::T) where {T <: Real}

    id_pos = findall(((μ .- y) .> 0) .& (u₁ .> 0))
    id_neg = findall(((y .- μ) .> 0) .& (u₂ .> 0))
    ids1 = id_pos[findall(log.((μ .- y[id_pos])./(σ*α)) .< 0)]
    ids2 = id_pos[findall(log.((μ .- y[id_pos])./(σ*α)) .> 0)]
    ids3 = id_neg[findall(log.((y[id_neg].-μ)./(σ*(1-α))) .< 0)]
    ids4 = id_neg[findall(log.((y[id_neg].-μ)./(σ*(1-α))) .> 0)]

    l1 = length(ids1) > 0 ? maximum(log.(u₁[ids1])./log.((μ .- y[ids1])./(α*σ))) : 0
    l2 = length(ids3) > 0 ? maximum(log.(u₂[ids3])./log.(( y[ids3] .- μ)./((1-α)*σ))) : 0

    up1 = length(ids2) > 0 ? minimum(log.(u₁[ids2]) ./ log.((μ .- y[ids2])./(α*σ))) : Inf
    up2 = length(ids4) > 0 ? minimum(log.(u₂[ids4]) ./log.((y[ids4] .- μ)./((1-α)*σ))) : Inf

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


function sampleθ(θ::T, ε::T,  y::Array{T, 1}, u₁::Array{T, 1}, u₂::Array{T, 1},
    μ::T, α::T, σ::T) where {T <: Real}
    interval = θinterval(y, u₁, u₂, μ, α, σ)
    prop = rand(Uniform(minimum(interval), maximum(interval)))
    θcond(prop, u₁, u₂, α) - θcond(θ, u₁, u₂, α) >= log(rand(Uniform(0,1), 1)[1]) ? prop : θ
end

function sampleβ(X::Array{T, 2}, y::Array{T, 1}, u₁::Array{T, 1}, u₂::Array{T, 1},
    β::Array{T, 1}, α::T, θ::T, σ::T, τ::T) where {T <: Real}
    n, p = size(X)
    βₛ = zeros(p)
    for k ∈ 1:p
        l, u = [], []
        for i ∈ 1:n
            μ = X[i, 1:end .!= k] ⋅  β[1:end .!= k]
            if X[i,k] > 0
                u₂[i] > 0 && append!(l, (y[i]-μ-(1-α)*σ*u₂[i]^(1/θ))/X[i,k])
                u₁[i] > 0 && append!(u, (y[i]-μ+ α*σ*u₁[i]^(1/θ))/X[i,k])
            elseif X[i, k] < 0
                u₁[i] > 0 && append!(l, (μ - y[i] - α*σ*u₁[i]^(1/θ))/(-X[i,k]))
                u₂[i] > 0 && append!(u, (μ - y[i] + (1-α)*σ*u₂[i]^(1/θ))/(-X[i,k]))
            end
        end
        length(l) == 0. && append!(l, -Inf)
        length(u) == 0. && append!(u, Inf)
        βₛ[k] = rand(truncated(Normal(0, τ), maximum(l), minimum(u[findall(u .>= maximum(l))])), 1)[1]
    end
    βₛ
end

function sampleμ(y::Array{T,1}, u₁::Array{T, 1}, u₂::Array{T, 1}, α::T, θ::T, σ::T, τ::T) where {T <: Real}
    n = length(y)
    l, u = [], []
    for i ∈ 1:n
        if u₁[i] > 0
            append!(u, y[i] + α*σ*u₁[i]^(1/θ))
        elseif u₂[i] > 0
            append!(l, y[i] - (1-α)*σ*u₂[i]^(1/θ))
        end
    end
    length(l) == 0. && append!(l, -Inf)
    length(u) == 0. && append!(u, Inf)
    maximum(l) < minimum(u) ? rand(truncated(Normal(0, τ), maximum(l), minimum(u)), 1)[1] : maximum(l)
end

function mcmc(y::Array{T, 1}, X::Array{T, 2}, α::T, nMCMC::N;
    θinit::T = 1., σinit::T = 1., printIter::N = 5000) where {T <: Real, N <: Integer}
    n = length(y)
    σ, σₗ, θ = zeros(nMCMC), zeros(nMCMC), zeros(nMCMC)
    σ[1] = σinit
    β = zeros(nMCMC, 2)
    β[1, :] = inv(X'*X)*X'*y
    θ[1] = θinit

    for i ∈ 2:nMCMC
        u1, u2 = sampleLatent(X, y, β[i-1,:], α, θ[i-1], σ[i-1])
        β[i,:] = sampleβ(X, y, u1, u2, β[i-1,:], α, θ[i-1], σ[i-1], 100.)
        σ[i], σₗ[i] = sampleσ(X, y, u1, u2, β[i, :], α, θ[i-1], 1, 1.)
        θ[i] = sampleθ(θ[i-1], .1, X, y, u1, u2, β[i, :], α, σ[i])
        if i % printIter === 0
            interval = θinterval(X, y, u1, u2, β[i,:], α, σ[i])
            printfmt("iter: {1}, θ ∈ [{2:.2f}, {3:.2f}], σ = {4:.2f} \n", i, interval[1], interval[2], σ[i])
        end
    end
    β, σ, σₗ, θ
end

function mcmc(y::Array{T, 1}, α::T, nMCMC::N; θinit::T = 1., printIter::N = 5000) where {T <: Real, N <: Integer}
    n = length(y)
    σ, μ, θ, σₗ = zeros(nMCMC), zeros(nMCMC), zeros(nMCMC), zeros(nMCMC)
    σ[1] = √var(y)
    μ[1] = mean(y)
    θ[1] = θinit

    for i ∈ 2:nMCMC
        u1, u2 = sampleLatent(y, μ[i-1], α, θ[i-1], σ[i-1])
        μ[i] = sampleμ(y, u1, u2, α, θ[i-1], σ[i-1], 100.)
        σ[i], σₗ[i] = sampleσ(y, u1, u2, μ[i], α, θ[i-1], 1, 1.)
        θ[i] = sampleθ(θ[i-1], .1, y, u1, u2, μ[i], α, σ[i])
        if i % 5000 === 0
            interval = θinterval(y, u1, u2, μ[i], α, σ[i])
            printfmt("iter: {1}, θ ∈ [{2:.2f}, {3:.2f}], σ = {4:.2f} \n", i, interval[1], interval[2], σ[i])
        end
    end
    μ, σ, σₗ, θ
end

# module
end
