include("QR.jl")
using .QR
using Distributions, LinearAlgebra, SpecialFunctions
using Plots, PlotThemes, KernelDensity
theme(:juno)

x = range(0.001, 0.5, length = 2000);
pdf.(InverseGamma(100, 1), x) |> (y-> plot(x, y))

## truncation point is around 1
n = 20;
β, α, θ, σ = [2.1, 0.8], 0.5, 2., 1.;
X = [repeat([1], n) rand(Uniform(-3, 3), n)];
y = X * β .+ rand(Laplace(0, σ), n);

u1, u2 = sampleLatent(X, y, β, α, θ, σ);

lower = zeros(n)
for i in 1:n
    μ = X[i,:] ⋅ β
    if (u1[i] > 0) && (y[i] < μ)
        lower[i] = (μ - y[i]) / (α * u1[i]^(1/θ))
    elseif (u2[i] > 0) && (y[i] >= μ)
        lower[i] = (y[i] - μ) / ((1-α) * u2[i]^(1/θ))
    end
end

maximum(lower)
