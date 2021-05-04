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

function A(x, p, α)::Real
    (1/2 + sign(x) * (1-α))^p
end

function δ(p, α)::Real
    2*α^p*(1-α)^p / (α^p + (1-α)^p)
end

function pdf(d::aepd, x::Real)
    del = δ(d.p, d.α)
    a = A(x-d.μ, d.p, d.α)
    C = del^(1/d.p) / (2^(1/d.p) * gamma(1+1/d.p))
    d.σ * C * exp(-0.5 * del/a * abs((x - d.μ)/d.σ)^d.p)
end

function rand(rng::AbstractRNG, d::aepd)
    del = δ(d.p, d.α)
    if rand(rng) < d.α
        d.μ + d.σ * 2^(1/d.p) * (-d.α * (rand(Gamma(1/d.p, 1))/del)^(1/d.p))
    else
        d.μ + d.σ * 2^(1/d.p) * ((1-d.α) * (rand(Gamma(1/d.p, 1))/del)^(1/d.p))
    end
end

function logpdf(d::aepd, x::Real)
    log(pdf(d, x))
end

end
