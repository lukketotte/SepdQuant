using StaticArrays, Distributions, LinearAlgebra, StatsBase, SpecialFunctions
include("../aepd.jl")
using .AEPD

n = 100;
β, α, σ = [2.1, 0.8], 0.5, 2.;
θ =  1.
X = ([repeat([1], n) rand(Uniform(10, 20), n)])
ϵ = rand(aepd(0., σ^(1/θ), θ, α), n);
y = X * β .+ ϵ

Xs = SMatrix{100,2}(X)
βs = SVector{2}(β)
ys = Xs*βs .+ SVector{n}(ϵ)

Xs[1,:] ⋅ β

function δ(α::Real, θ::Real)
    return 2*(α*(1-α))^θ / (α^θ + (1-α)^θ)
end


z = y - X*β
@time pos = z.>0
@time neg = z.<0
z[pos]

@time z[z.>0]
@time filter(x->x>0.,z)
@time findall(z.>0)

typeof(βs) <: SVector{2,Float64}
typeof(Xs) <: SArray{Tuple{100,2},Float64}

function ∇(β::MixedVector, X::MixedVector, y::MixedVector, α::Real, θ::Real, σ::Real, τ::Real)
    z = y - X*β
    pos, neg = z.>0, z.<0
    p = length(β)
    ∇ = MVector{p}(zeros(p))
    for k in 1:p
        ℓ₁ = θ/α^θ * sum((.-z[z.<0]).^(θ-1) .* X[z.<0, k])
        ℓ₂ = θ/(1-α)^θ * sum(z[z.>=0].^(θ-1) .* X[z.>=0, k])
        ∇[k] = -δ(α,θ)/σ * (ℓ₁ - ℓ₂) - β[k]/(τ^2)
    end
    return ∇
end

MixedVector = Union{SVector, Array{<:Real, 1}}
size(Xs)

typeof(βs) <: MixedVector
typeof(ys)
@time ∇(βs, Xs, ys, 0.5, 1., 1., 100)
@time ∇(β, X, y, 0.5, 1., 1., 100)

typeof(ys)
