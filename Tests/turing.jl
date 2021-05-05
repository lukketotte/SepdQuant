using Turing
using StatsPlots
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

c1 = sample(gdemo([1.5, 1.1, 3.2], 2), SMC(), 1000)
plot(c1)

@model function aepdmcmc(x)
    α ~ Uniform(0, 1)
    for i in eachindex(x)
        x[i] ~ aepd(0., 1., 3., α)
    end
end

d = aepd(0., 1., 3., 0.5);
x = rand(d, 1000);
√var(x)

c1 = sample(aepdmcmc(x), SMC(), 1000)

plot(c1)
