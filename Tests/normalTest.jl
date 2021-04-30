using Distributions, LinearAlgebra

## Testing data
y = rand(Normal(0, 2), 50)

## helpers
function ESS(W::Vector{T})::T where {T <: Real}
    1 / sum(W.^2)
end

function α₁(Tₖ::N, t::N; p::T = 2.) where {N <: Integer, T <: Real}
    (t/Tₖ)^p
end



## initialization step
N = 100
θ = zeros(N, 2)
T = 100
# hyperparameters
μ₀, σ₀ = 0, 10
α₀, β₀ = 1, 2

# initialize using ν ~ π(θ)
σ = 1 ./ rand(Gamma(α₀,β₀), N)
μ = zeros(N)
for i in 1:N
    μ[i] = rand(Normal(μ₀, √(σ₀ / σ[i])), 1)[1]
end

θ = hcat(μ, σ)

# weights
W = zeros(N, T)
W[:, 1] .= 1/N

prod(pdf.(Normal(μ[3], σ[3]), y)) ^ α₁(T, 1)

for i in 1:N
    W[i, 2] = W[i, 1] * prod(pdf.(Normal(μ[i], σ[i]), y)) ^ α₁(T, 1)
end

sum(W[:, 2].^2)

ESS(W[:, 2])

# step t
for t in 2:T
    for i in 1:N
        lik = pdf.(Normal())
    end
end
