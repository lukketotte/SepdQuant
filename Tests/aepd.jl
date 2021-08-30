module AEPD

export aepd

using Distributions, LinearAlgebra, SpecialFunctions, Random
import Base.rand, Distributions.pdf, Distributions.logpdf

struct aepd <: ContinuousUnivariateDistribution
    μ::Real
    σ::Real
    p::Real
    α::Real
    aepd(μ, σ, p, α) = new(Real(μ), Real(σ), Real(p), Real(α))
end

params(d::aepd) = (d.μ, d.σ, d.p, d.α)

function δ(p, α)::Real
    2*α^p*(1-α)^p / (α^p + (1-α)^p)
end

"""function pdf(d::aepd, x::Real)
    del = δ(d.p, d.α)
    C = del^(1/d.p) / (gamma(1+1/d.p) * d.σ^(1/d.p))
    x < d.μ ? C * exp(- del/(d.σ*d.α) * (d.μ-x)^d.p) : C * exp(- del/(d.σ*(1-d.α)) * (x-d.μ)^d.p)
end

function logpdf(d::aepd, x::Real)
    log(pdf(d, x))
end"""


function logpdf(d::aepd, x::Real)
    μ, σ, p, α = params(d)
    del = δ(p, α)
    C = del^(1/p) / (gamma(1+1/p) * σ^(1/p))
    x < μ ? log(C) - del/(σ*α) * (μ-x)^p : log(C) - del/(σ*(1-α)) * (x-μ)^p
end

pdf(d::aepd, x::Real) = exp(logpdf(d, x))

function rand(rng::AbstractRNG, d::aepd)
    if rand(rng) < d.α
        d.μ + d.σ * (-d.α * (rand(Gamma(1/d.p, 1))/δ(d.p, d.α))^(1/d.p))
    else
        d.μ + d.σ * ((1-d.α) * (rand(Gamma(1/d.p, 1))/δ(d.p, d.α))^(1/d.p))
    end
end

end
