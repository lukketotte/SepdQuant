## Code for review
using Distributed, SharedArrays
@everywhere using SepdQuantile, LinearAlgebra, StatsBase, QuantileRegressions, DataFrames, Distributions, RCall

@everywhere function bayesQR(y::AbstractVector{<:Real}, X::AbstractMatrix{<:Real}, quant::Real, ndraw::Int, keep::Int)
  rcopy(R"""
      suppressWarnings(suppressMessages(library(bayesQR, lib.loc = "C:/Users/lukar818/Documents/R/win-library/4.0")))
      X = $X
      y = $y
      quant = $quant
      ndraw = $ndraw
      keep = $keep
      bayesQR(y ~ X[,2], quantile = quant, ndraw = ndraw, keep=keep)[[1]]$betadraw
  """)
end


control =  Dict(:tol => 1e-3, :max_iter => 1000, :max_upd => 0.3,
  :is_se => false, :est_beta => true, :est_sigma => true,
  :est_p => true, :est_tau => true, :log => false, :verbose => false)

n = 200
X = hcat(ones(n), rand(Normal(), n), rand(Normal(), n))
α = 0.05
B = 500
reps = 10
boots = SharedArray(zeros(Float64, (reps, B, 2)));

@sync @distributed for r in 1:reps
  println("rep: $r")
  ϵ, τ = bivmix(n,  0.88089, -2.5, 1, 0.5, 1.), 0.9
  # ϵ, τ = rand(Normal(-1.281456, 1), n), 0.9
  # ϵ, τ = rand(NoncentralT(6, 1.2815281528152815), n), 0.1
  # ϵ, τ = (1 .+ X[:,2]).*rand(Normal(), n), 0.5
  y = X*ones(3) + ϵ
  for b in 1:B
    b % 10 == 0 && println(b)
    ids = sample(1:n, n)
    control[:est_sigma], control[:est_tau], control[:est_p] = (true, true, true)
    try
      res = quantfreq(y[ids], X[ids,:], control, 2., 1.)
      taufreq = mcτ(τ, res[:tau], res[:p], res[:sigma])
      control[:est_sigma], control[:est_tau], control[:est_p] = (false, false, false)
      res = quantfreq(y[ids], X[ids,:], control, res[:sigma], res[:p], taufreq, 1.)
      boots[r,b,1] = res[:beta][2]
    catch e
      println("error")
      boots[r,b,1] = boots[r,b-1, 1]
    end
    bqr = DataFrame(hcat(y[ids], X[ids,:]), :auto) |> x -> qreg(@formula(x1 ~  x3 + x4), x, τ) |> coef
    boots[r,b,2] = bqr[2]
  end
end

[quantile(boots[i,:,1], α/2) <= 1. <= quantile(boots[i,:,1], 1-α/2) for i in 1:reps] |> mean
[quantile(boots[i,:,2], α/2) <= 1. <= quantile(boots[i,:,2], 1-α/2) for i in 1:reps] |> mean

### cross validation
using MLDataUtils

K = 5
n = length(y)
folds = kfolds(sample(1:n, n; replace = false), k = K)

bqr = SharedArray(zeros(Float64, K))
fqr = SharedArray(zeros(Float64, K))
bsqr = SharedArray(zeros(Float64, K))
fsqr = SharedArray(zeros(Float64, K))

τ = 0.2

@sync @distributed for k in 1:K
  Xtest, ytest = X[folds[k][2],:], y[folds[k][2]]
  Xtrain, ytrain = X[Not(folds[k][2]),:], y[Not(folds[k][2])]

  # Freq
  control[:est_sigma], control[:est_tau], control[:est_p] = (true, true, true)
  res = quantfreq(ytrain, Xtrain, control, 2., 1.)
  taufreq = mcτ(τ, res[:tau], res[:p], res[:sigma])
  control[:est_sigma], control[:est_tau], control[:est_p] = (false, false, false)
  res = quantfreq(ytrain, Xtrain, control, res[:sigma], res[:p], taufreq)
  fsqr[k] = mean(ytest .<= Xtest * res[:beta])

  # Bayesian
  par = Sampler(ytrain, Xtrain, 0.5, 12000, 5, 2000);
  β, θ, σ, α = mcmc(par, .3, 0.11, 1.1, 2, 1, 0.5, rand(size(par.X, 2)))
  par.α = mcτ(τ, mean(α), mean(θ), mean(σ))
  βlp = mcmc(par, .25, mean(θ), mean(σ), [0.6, 0.8]; verbose = false)
  blp = permutedims(mean(βlp, dims = 1))
  bsqr[k] = mean(ytest .<= Xtest * blp)
  # Classical
  #βqr = DataFrame(hcat(par.y, par.X), :auto) |> x -> qreg(@formula(x1 ~  x3), x, τ) |> coef
  #fqr[k] = mean(ytest .<= Xtest * βqr)
end

# take this one seperately..
@sync @distributed for k in 1:K
  Xtest, ytest = X[folds[k][2],:], y[folds[k][2]]
  Xtrain, ytrain = X[Not(folds[k][2]),:], y[Not(folds[k][2])]
  anyNan = true
  while anyNan
    β = bayesQR(ytrain, Xtrain, 0.4, 10000, 4)
    anyNan = any(isnan.(β)) ? true : false
  end
  baqr = permutedims(mean(β, dims = 1))
  bqr[k] = mean(ytest .<= Xtest * baqr)
end

mean(bqr)
√var(bqr)

mean(fqr)
√var(fqr)

mean(bsqr)
√var(bsqr)

mean(fsqr)
√var(fsqr)