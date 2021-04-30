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
N = 10
θ = zeros(N, 2)
T = 100
# hyperparameters
μ₀, σ₀ = 0, 10
α₀, β₀ = 1, 2

# initialize using ν ~ π(θ)
σ = 1 ./ rand(Gamma(α₀,β₀), N)
μ = zeros(N)
for i in 1:N
    μ[i] = rand(Normal(μ₀, σ₀ / σ[i]), 1)[1]
end

θ = hcat(μ, σ)

# weights
W = zeros(N, T)
W[:, 1] .= 1
