#include("../../Distributions.jl/src/Distributions.jl")
using KernelDensity, Plots, SpecialFunctions, Distributions
include("sepd.jl")
using .SEPD

p = 2
K = 2*p^(1/p)*gamma(1+1/p)
d = SkewedExponentialPower(0, 1, 2, 0.5)
x = rand(d, 10000)
k = kde(x)
y = range(minimum(x), maximum(x), length = 100)
plot(y, pdf(k, y))
plot!(y, pdf.(SkewedExponentialPower(0,1,2,0.5), y))
