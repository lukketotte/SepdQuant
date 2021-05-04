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

c1 = sample(gdemo(1.5, 2), SMC(), 1000)
plot(c1)

@model function aepdmcmc(x)
    α ~ Uniform(0, 1)
    x ~ aepd(0., 1., 2., α)
end

d = aepd(0., 1., 2., 0.8);
x = rand(d, 100);

c1 = sample(aepdmcmc([-2. 2.]), SMC(), 1000)

plot(c1)
