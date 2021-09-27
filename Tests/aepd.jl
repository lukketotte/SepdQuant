module AEPD

export Aepd

using Distributions, LinearAlgebra, SpecialFunctions, Random
import Base.rand
import Distributions: pdf, logpdf, @check_args

struct Aepd{T <: Real} <: ContinuousUnivariateDistribution
    μ::T
    σ::T
    p::T
    α::T
    Aepd{T}(μ::T, σ::T, p::T, α::T) where {T} = new{T}(μ, σ, p, α)
end

function Aepd(µ::T, σ::T, p::T, α::T; check_args=true) where {T <: Real}
    check_args && @check_args(Aepd, σ > zero(σ))
    check_args && @check_args(Aepd, p > zero(p))
    check_args && @check_args(Aepd, α > zero(α) && α < one(α))
    return Aepd{T}(µ, σ, p, α)
end

Aepd(μ::Real, σ::Real, p::Real, α::Real) = Aepd(promote(μ, σ, p, α)...)

params(d::Aepd) = (d.μ, d.σ, d.p, d.α)

function δ(p, α)::Real
    2*α^p*(1-α)^p / (α^p + (1-α)^p)
end

function logpdf(d::Aepd, x::Real)
    μ, σ, p, α = params(d)
    del = δ(p, α)
    C = del^(1/p) / (gamma(1+1/p) * σ^(1/p))
    x < μ ? log(C) - del/(σ*α^p) * (μ-x)^p : log(C) - del/(σ*(1-α)^p) * (x-μ)^p
end

pdf(d::Aepd, x::Real) = exp(logpdf(d, x))

function rand(rng::AbstractRNG, d::Aepd)
    if rand(rng) < d.α
        d.μ + d.σ * (-d.α * (rand(Gamma(1/d.p, 1))/δ(d.p, d.α))^(1/d.p))
    else
        d.μ + d.σ * ((1-d.α) * (rand(Gamma(1/d.p, 1))/δ(d.p, d.α))^(1/d.p))
    end
end

end
