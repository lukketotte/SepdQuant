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

Aepd(μ::Real, σ::Real, p::Real, α::Real) = Aepd( promote(μ, σ, p, α)...)

params(d::Aepd) = (d.μ, d.σ, d.p, d.α)


function logpdf(d::Aepd, x::Real)
    μ, σ, p, α = params(d)
    -log(σ) - gamma(1+1/p)^p * (x < μ ? ((μ-x)/(α*σ))^p : ((x-μ)/((1-α)*σ))^p)
end

pdf(d::Aepd, x::Real) = exp(logpdf(d, x))

function rand(rng::AbstractRNG, d::Aepd)
    μ, σ, p, α = params(d)
    if rand(rng) < d.α
        μ - σ * α * rand(Gamma(1/p, 1))^(1/p) / (gamma(1+1/p))
    else
        μ + σ * (1-α) * rand(Gamma(1/p, 1))^(1/p) / (gamma(1+1/p))
    end
end

end
