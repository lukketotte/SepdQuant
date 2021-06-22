using Distributions, LinearAlgebra, StatsBase, SpecialFunctions
include("QR.jl")
include("../aepd.jl")
using .AEPD, .QR
using Plots, PlotThemes, CSV, DataFrames, StatFiles


## Struct test
struct MCMCparams
    y::Array{<:Real, 1}
    X::Array{<:Real, 2}
    nMCMC::Int
    thin::Int
    burnIn::Int
    ε::Union{Real, Array{<:Real, 1}}
    Θ::Union{Array{Any, 2}}

    function MCMCparams(y::Array{<:Real, 1}, X::Array{<:Real, 2}, nMCMC::Int,
            thin::Int, burnIn::Int, ε::Union{Real, Array{<:Real, 1}})
        new(y, X, nMCMC, thin, burnIn, ε, [[inv(X'*X)*X'*y] [1.] [1.]])
    end

    function MCMCparams(y::Array{<:Real, 1}, X::Array{<:Real, 2}, nMCMC::Int,
            thin::Int, burnIn::Int)
        new(y, X, nMCMC, thin, burnIn, 0.01, [[inv(X'*X)*X'*y] [1.] [1.]])
    end
end

@showprogress 1 "Computing..." for i in 1:500000000
    log(i)
end


n = 500;
β, α, σ = [2.1, 0.8], 0.5, 2.;
θ =  1.
X = [repeat([1], n) rand(Uniform(10, 20), n)]
y = X*β .+ rand(Laplace(0, 1), n)
