module SEPD

export SkewedExponentialPower

using Distributions, SpecialFunctions, Random
import Base.rand, Base.sign
import Distributions: pdf, logpdf, @check_args, partype, @distr_support, cdf
import StatsBase: kurtosis, skewness, entropy, mode, modes,
                  fit, kldivergence, loglikelihood, dof, span,
                  params, params!
import Statistics: mean, median, quantile, std, var, cov, cor

"""
    SkewExponentialPower(μ, σ, p, α)
The *Skewed exponential power distribution*, with location `μ`, scale `σ`, shape `p`, and skewness `α`
has the probability density function
```math
f(x; \\mu, \\sigma, \\p, \\alpha) =
\\begin{cases}
\\frac{1}{\\sigma}K_{EP}(p_1) \\exp \\left\\{ - \\frac{1}{2p}\\Big| \\frac{x-\\mu}{\\alpha \\sigma} \\Big|^{p_1} \\right\\}, & \\text{if } x \\leq \\mu; \\\\
\\frac{1}{\\sigma}K_{EP}(p_2) \\exp \\left\\{ - \\frac{1}{2p}\\Big| \\frac{x-\\mu}{(1-\\alpha) \\sigma} \\Big|^{p_2} \\right\\}, & \\text{if } x > \\mu,
\\end{cases}
```
where ``K_{EP}(p) = 1/[2p^{1/p}\\Gamma(1+1/p)]``.
The Skewed exponential power distribution (SEPD) incorporates the laplace (``p=1, \\alpha=0.5``),
normal (``p=2, \\alpha=0.5``), uniform (``p\\rightarrow \\infty, \\alpha=0.5``), asymmetric laplace (``p=1``), skew normal (``p=2``),
and exponential power distribution (``\\alpha = 0.5``) as special cases.
```julia
SkewExponentialPower()            # SEPD with shape 2, scale 1, location 0, and skewness 0.5 (the standard normal distribution)
SkewExponentialPower(μ, σ, p, α)  # SEPD with location μ, scale σ, shape p, and skewness α
SkewExponentialPower(μ, σ, p)     # SEPD with location μ, scale σ, shape p, and skewness 0.5 (the exponential power distribution)
SkewExponentialPower(μ, σ)        # SEPD with location μ, scale σ, shape 2, and skewness 0.5 (the normal distribution)
SkewExponentialPower(σ)           # SEPD with location 0, scale σ, shape 2, and skewness 0.5 (the normal distribution)
params(d)       # Get the parameters, i.e. (μ, σ, p, α)
shape(d)        # Get the shape parameter, p
skewness(d)     # Get the skewness parameter, α
location(d)     # Get the location parameter, μ
```
"""
struct SkewedExponentialPower{T <: Real} <: ContinuousUnivariateDistribution
    μ::T
    σ::T
    p::T
    α::T
    SkewedExponentialPower{T}(μ::T, σ::T, p::T, α::T) where {T} = new{T}(μ, σ, p, α)
end

function SkewedExponentialPower(µ::T, σ::T, p::T, α::T; check_args=true) where {T <: Real}
    if check_args
        @check_args(SkewedExponentialPower, σ > zero(σ))
        @check_args(SkewedExponentialPower, p > zero(p))
        @check_args(SkewedExponentialPower, zero(α) < α < one(α))
    end
    return SkewedExponentialPower{T}(µ, σ, p, α)
end

function SkewedExponentialPower(μ::Real=0, σ::Real=1, p::Real=2, α::Real=1//2; kwargs...)
    return SkewedExponentialPower(promote(μ, σ, p, α)...; kwargs...)
end

@distr_support SkewedExponentialPower -Inf Inf

### Conversions
convert(::Type{SkewedExponentialPower{T}}, μ::S, σ::S, p::S, α::S) where {T <: Real, S <: Real} = SkewedExponentialPower(T(μ), T(σ), T(p), T(α))
convert(::Type{SkewedExponentialPower{T}}, d::SkewedExponentialPower{S}) where {T <: Real, S <: Real} = SkewedExponentialPower(T(d.μ), T(d.σ), T(d.p), T(d.α), check_args=false)
convert(::Type{SkewedExponentialPower{T}}, d::SkewedExponentialPower{T}) where {T<:Real} = d

### Parameters
@inline partype(d::SkewedExponentialPower{T}) where {T<:Real} = T

params(d::SkewedExponentialPower) = (d.μ, d.σ, d.p, d.α)
location(d::SkewedExponentialPower) = d.μ
shape(d::SkewedExponentialPower) = d.p
scale(d::SkewedExponentialPower) = d.σ

### Statistics

#Calculates the kth central moment of the SEPD
function m_k(d::SkewedExponentialPower, k::Integer)
    _, σ, p, α = params(d)
    inv_p = inv(p)
    return  ((k*log(2) + k*inv(p) * log(p) + k*log(σ) + loggamma((1+k)*inv(p)) -
        loggamma(inv(p))) + log(abs((-1)^k*α^(1+k) + (1-α)^(1+k))))
end

# needed for odd moments on log-scale
sign(d::SkewedExponentialPower) = d.α > 0.5 ? -1 : 1

mean(d::SkewedExponentialPower) = d.α == 0.5 ? d.μ : sign(d)*exp(m_k(d, 1)) + d.μ
mode(d::SkewedExponentialPower) =  mean(d)
var(d::SkewedExponentialPower) = exp(m_k(d, 2)) - exp(2*m_k(d, 1))
skewness(d::SkewedExponentialPower) = d.α == 1//2 ? float(zero(partype(d))) : sign(d)*exp(m_k(d, 3)) / (std(d))^3
kurtosis(d::SkewedExponentialPower) = exp(m_k(d, 4))/var(d)^2 - 3

function logpdf(d::SkewedExponentialPower, x::Real)
    μ, σ, p, α = params(d)
    a = x < μ ? α : 1 - α
    inv_p = inv(p)
    return -(log(2) + log(σ) + inv_p * log(p) + loggamma(1 + inv_p) + inv_p * (abs(μ - x) / (2 * σ * a))^p)
end

function cdf(d::SkewedExponentialPower, x::Real)
    μ, σ, p, α = params(d)
    inv_p = inv(p)
    if x <= μ
        α * ccdf(Gamma(inv_p), inv_p * (abs((x-μ)/σ) / (2*α))^p)
    else
        α + (1-α) * cdf(Gamma(inv_p), inv_p * (abs((x-μ)/σ) / (2*(1-α)))^p)
    end
end

function quantile(d::SkewedExponentialPower, p::Real)
    μ, σ, _, α = params(d)
    inv_p = inv(d.p)
    if p <= α
        μ - 2*α*σ * (d.p * quantile(Gamma(inv_p), 1-p/α))^(inv_p)
    else
        μ + 2*(1-α)*σ * (d.p * quantile(Gamma(inv_p), 1-(1-p)/(1-α)))^(inv_p)
    end
end

function rand(rng::AbstractRNG, d::SkewedExponentialPower)
    μ, σ, p, α = params(d)
    inv_p = inv(d.p)
    if rand(rng) < d.α
        μ - σ * 2*p^(inv_p) * α * rand(Gamma(inv_p, 1))^(inv_p)
    else
        μ + σ * 2*p^(inv_p) * (1-α) * rand(Gamma(inv_p, 1))^(inv_p)
    end
end


end
