using Distributed, SharedArrays
@everywhere using SepdQuantile, LinearAlgebra, StatsBase, QuantileRegressions, DataFrames, Distributions

control =  Dict(:tol => 1e-3, :max_iter => 1000, :max_upd => 0.3,
  :is_se => false, :est_beta => true, :est_sigma => true,
  :est_p => true, :est_tau => true, :log => false, :verbose => false)

n = 200
X = hcat(ones(n), rand(Normal(), n), rand(Normal(), n))
τ = 0.9
α = 0.05
B = 500
reps = 200
boots = SharedArray(zeros(Float64, (reps, B, 2)));

@sync @distributed for r in 1:reps
  println("rep: $r")
  #ϵ = bivmix(n,  1-0.88089, -1, 2.5, 1, 0.5)
  ϵ = rand(Normal(-1.281456, 1), n)
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

# 5: 1.0, 1.0
# 10: 1.0, 1.0

quantile(boots[1,:,1], α/2) <= 1. <= quantile(boots[1,:,1], 1-α/2)
quantile(boots[1,:,2], α/2) <= 1. <= quantile(boots[1,:,2], 1-α/2)

mean(boots; dims = 1)

using StatsPlots
density(boots[:,1])
density!(boots[:,2])
