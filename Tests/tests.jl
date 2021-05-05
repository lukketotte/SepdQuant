include("aepd.jl")
using .AEPD
using Distributions, LinearAlgebra, Plots, PlotThemes, KernelDensity
theme(:juno)

x = range(-5, 5, length = 500);
d = aepd(0., 1., 2., 0.5);
k = kde(rand(d, 10000));
plot(x, pdf(k, x))

√var(rand(d, 10000))
