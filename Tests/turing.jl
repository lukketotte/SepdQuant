using Turing, StatsPlots
include("aepd.jl")
using .AEPD
theme(:juno)

# Define a simple Normal model with unknown mean and variance.
@model function gdemo(x, y)
    s ~ InverseGamma(2, 3)
    m ~ Normal(0, sqrt(s))
    x ~ Normal(m, sqrt(s))
    y ~ Normal(m, sqrt(s))
end

c1 = sample(gdemo(1.5, 2), SMC(), 1000)
plot(c1)

@model function aepdmcmc(x)
    α ~ Uniform(0.5, 1)
    p ~ Uniform(0.5, 3)
    μ ~ Normal(0, 10)
    # σ ~ InverseGamma(1, 1)
    for i in eachindex(x)
        x[i] ~ aepd(μ, 1., p, α)
    end
end

# Bijectors.bijector(d::aepd) = Logit(0., 1.)
d = aepd(0., 1., 1., 0.7);
x = rand(d, 300);


c1 = sample(aepdmcmc(x), Gibbs(PG(10, :α), PG(10, :p), PG(10, :μ)), 1000)
plot(c1)
