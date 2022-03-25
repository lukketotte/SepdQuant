using Distributed, SharedArrays
@everywhere using Distributions, LinearAlgebra, StatsBase, SpecialFunctions, QuantileRegressions, DataFrames, QuadGK, RCall
@everywhere include(joinpath(@__DIR__, "../aepd.jl"))
@everywhere include(joinpath(@__DIR__, "../../QuantileReg/QuantileReg.jl"))
@everywhere include(joinpath(@__DIR__, "../../QuantileReg/FreqQuantileReg.jl"))
@everywhere using .AEPD, .QuantileReg, .FreqQuantileReg

using Plots, PlotThemes, CSV, StatFiles, CSVFiles, HTTP
theme(:juno)
using Formatting

## α to τ
@everywhere f(x, b, p, α, μ, σ) = abs(x-b)^(p-1) * pdf(Aepd(μ, σ, p, α), x)

@everywhere function quantconvert(q, p, α, μ, σ)
    a₁ = quadgk(x -> f(x, q, p, α, μ, σ), -Inf, Inf)[1]
    a₂ = quadgk(x -> f(x, q, p, α, μ, σ), -Inf, q)[1]
    1/((maximum([a₁/a₂, 1.0001]) - 1)^(1/p) + 1)
end

@everywhere function mcτ(τ, α, p, σ, n = 1000, N = 1000)
    res = zeros(N)
    for i in 1:N
        dat = rand(Aepd(0, σ, p, α), n)
        q = DataFrame(hcat(dat), :auto) |> x -> qreg(@formula(x1 ~  1), x, τ) |> coef;
        #res[i] = quantconvert(q[1], p, α, 0, σ)
        a₁ = mean(abs.(dat .- q).^(p-1))
        a₂ = mean(abs.(dat .- q).^(p-1) .* (dat .< q))
        res[i] = 1/((maximum([a₁/a₂, 1.0001]) - 1)^(1/p) + 1)
    end
    mean(res)
end

## RMSE & bias comparison
N = 100
res = SharedArray(zeros((N, 2)))
resQuant = SharedArray(zeros((N, 2)))
n = 200
X = hcat(ones(n), rand(Normal(), n), rand(Normal(), n))

@sync @distributed for i ∈ 1:N
    println(i)
    #n = 500;
    #X = hcat(ones(n), rand(Normal(), n), rand(Normal(), n))
    y = X*ones(3) + rand(Chisq(3), n)#rand(TDist(3), n)

    par = Sampler(y, X, 0.5, 12000, 5, 4000)
    β, θ, σ, α = mcmc(par, 0.5, 0.5, 1., 1, 2, 0.5, [0., 0., 0.]; verbose = false)
    par.α = mcτ(0.9, mean(α), mean(θ), mean(σ))

    βlp = mcmc(par, .1, mean(θ), mean(σ), [0., 0., 0.]; verbose = false)
    blp = mean(βlp, dims = 1)'
    bqr = DataFrame(hcat(par.y, par.X), :auto) |> x -> qreg(@formula(x1 ~  x3 + x4), x, 0.9) |> coef

    res[i, 1] = mean((blp-ones(3)).^2)
    res[i, 2] = mean((bqr-ones(3)).^2)
    resQuant[i, 1] = mean(par.y .<= par.X * blp)
    resQuant[i, 2] = mean(par.y .<= par.X * bqr)
end

res

mean(resQuant, dims = 1)
mean(sqrt.(res), dims = 1)
√var(res[:,1])
√var(res[:,2])

res2 = res

## Application, max temp in Melbourne?
dat = HTTP.get("https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-max-temperatures.csv") |> x -> CSV.File(x.body) |> DataFrame
y = log.(dat[2:size(dat, 1),:Temperature])
X = hcat(ones(length(y)), log.(dat[1:(size(dat,1)-1),:Temperature]))

settings = DataFrame(tau = range(0.1, 0.9, length = 9), old = 0, bayes = 0, freq = 0, qr = 0)

par = Sampler(y, X, 0.5, 10000, 1, 1000);
β, θ, σ, α = mcmc(par, .25, 0.15, 1., 2, 1, 0.5, rand(size(par.X, 2)));
μ = X * [median(β[:, i]) for i in 1:size(X, 2)]
res = quantfreq(y, X, control);
μf = X * res[:beta]
control[:est_sigma], control[:est_tau], control[:est_p] = (false, false, false)

settings = SharedArray(Matrix(settings))

#CSV.write("mcmc.csv", DataFrame(shape = θ, scale = σ, skewness = α, beta = β[:,2]))

@sync @distributed for i ∈ 1:size(settings, 1)
    b = DataFrame(hcat(y, X), :auto) |> x -> qreg(@formula(x1 ~  x3), x, settings[i,1]) |> coef
    q = X * b
    settings[i, 5] = mean(y .<= q)

    if 0.1 < settings[i,1] < 0.9
        par.α = [quantconvert(q[j], mean(θ), mean(α), μ[j], mean(σ)) for j in 1:length(y)] |> mean
        convTau = [quantconvert(q[j], res[:p], res[:tau], μf[j], res[:sigma]) for j in 1:length(y)] |> mean
    else
        par.α = mcτ(settings[i,1], mean(α), mean(θ), mean(σ), 5000)
        convTau = mcτ(settings[i,1], res[:tau], res[:p], res[:sigma], 5000)
    end

    βres = mcmc(par, 0.5, mean(θ), mean(σ), rand(size(par.X, 2)))
    settings[i, 3] = mean(par.y .<= par.X *  median(βres, dims = 1)')

    temp = quantfreq(y, X, control, res[:sigma], res[:p], convTau)
    settings[i,4] = mean(y .<= X*temp[:beta])

    par.α = settings[i,1]
    β1, _, _ = mcmc(par, .25, 1., 1., 2, rand(size(par.X, 2)))
    settings[i, 2] = mean(par.y .<= par.X *  median(β1, dims = 1)')
end
settings
plt_dat = DataFrame(Tables.table(settings)) |> x -> rename!(x, ["tau", "old", "bayes", "freq", "qr"])
CSV.write("C:/Users/lukar818/Dropbox/PhD/research/applied/quantile/R/plots/tempquant.csv", plt_dat)

## Simulation study with AEPD error term
n = 350;
x = rand(Normal(), n);
X = hcat(ones(n), x)

p = [1.5, 2., 2.5]
skew = [0.1, 0.5, 0.9]
quant = [0.1, 0.5, 0.9]
# quant = range(0.1, 0.9, length = 3)

settings = DataFrame(p = repeat(p, inner = length(skew)) |> x -> repeat(x, inner = length(quant)),
    skew = repeat(skew, length(p)) |> x -> repeat(x, inner = length(quant)),
    tau = repeat(quant, length(skew) * length(p)), old = 0,  sdOld = 0, bayes = 0,
    sdBayes = 0, freq = 0, sdFreq = 0)

cols = names(settings)
settings = SharedArray(Matrix(settings))
reps = 10

control =  Dict(:tol => 1e-3, :max_iter => 1000, :max_upd => 0.3,
  :is_se => false, :est_beta => true, :est_sigma => true,
  :est_p => true, :est_tau => true, :log => false, :verbose => false)

@sync @distributed for i ∈ 1:size(settings, 1)
    println(i)
    p, skew, τ = settings[i, 1:3]
    old, bayes, freq = [zeros(reps) for i in 1:3]
    for j ∈ 1:reps
        y = 2.1 .+ 0.5 .* x + rand(Aepd(0, 1, p, skew), n);

        # Bayesian
        par = Sampler(y, X, skew, 10000, 5, 2500);
        β, θ, σ, α = mcmc(par, 0.8, .25, 1.5, 1, 2, 0.5, [2.1, 0.5], verbose = false);
        μ = X * median(β, dims = 1)' |> x -> reshape(x, size(x, 1))

        # Freq
        control[:est_sigma], control[:est_tau], control[:est_p] = (true, true, true)
        res = quantfreq(y, X, control)
        μf = X * res[:beta] |> x -> reshape(x, size(x, 1))

        # Compute τ converted
        b = DataFrame(hcat(par.y, par.X), :auto) |> x -> qreg(@formula(x1 ~  x3), x, τ) |> coef
        q = X * b
        if n >= 350
            taubayes = [quantconvert(q[k], median(θ), median(α), μ[k], median(σ)) for k in 1:length(par.y)] |> mean
            taufreq  = [quantconvert(q[k], res[:p], res[:tau], μf[k], res[:sigma]) for k in 1:length(y)] |> mean
        else
            taubayes = mcτ(α[i], median(α), median(θ), median(σ), 2500)
            taufreq  = mcτ(α[i], res[:tau], res[:p], res[:sigma], 2500)
        end

        # Compute estimated quantiles based on conversion
        par.α = taubayes
        βres = mcmc(par, 1.3, median(θ), median(σ), b, verbose = false)
        μ = X * median(βres, dims = 1)' |> x -> reshape(x, size(x, 1))
        bayes[j] = [par.y[k] <= μ[k]  for k in 1:length(par.y)] |> mean

        par.α = τ
        par.πθ = "uniform"
        βt, _, _ = mcmc(par, .6, .6, 1.2, 2, b, verbose = false)
        μ = X * median(βt, dims = 1)' |> x -> reshape(x, size(x, 1))
        old[j] = [par.y[k] <= μ[k]  for k in 1:length(par.y)] |> mean

        control[:est_sigma], control[:est_tau], control[:est_p] = (false, false, false)
        res = quantfreq(y, X, control, res[:sigma], res[:p], taufreq)
        freq[j] = mean(y .<= X*res[:beta])

    end
    settings[i, 4] = mean(old)
    settings[i, 5] = √var(old)
    settings[i, 6] = mean(bayes)
    settings[i, 7] = √var(bayes)
    settings[i, 8] = mean(freq)
    settings[i, 9] = √var(freq)
end

plt_dat = DataFrame(Tables.table(settings)) |> x -> rename!(x, cols)
CSV.write("C:/Users/lukar818/Dropbox/PhD/research/applied/quantile/R/plots/simulations/sims250_9.csv", plt_dat)

plt_dat1[:, 4:9]

test = (plt_dat1[:, :old] + plt_dat2[:, :old])/2

(plt_dat1[2, :old]+plt_dat2[2,:old])/2 === test[2]






# simulation with other random errors
quant = [0.1, 0.5, 0.9]

#dists = ["Gumbel", "Erlang", "Tdist", "Chi"]
dists = [1,2,3,4]

settings = DataFrame(tau = repeat(quant, length(dists)), dist = repeat(dists, inner = length(quant)),
    bayes = 0, sdBayes = 0, freq = 0, sdFreq = 0, old = 0, sdOld = 0)


cols = names(settings)
settings = SharedArray(Matrix(settings))
reps = 20

@sync @distributed for i ∈ 1:size(settings, 1)
    old, bayes, freq = [zeros(reps) for i in 1:3]
    for j ∈ 1:reps
        if settings[i, 2] == 1 #"Gumbel"
            y = 2.1 .+ 0.5 .* x + rand(Gumbel(0, 1), n)
            ε = [0.6, 1]
        elseif settings[i, 2] == 2#"Erlang"
            y = 2.1 .+ 0.5 .* x + rand(Erlang(7, 0.5), n)
            ε = [0.6, 1]
        elseif settings[i, 2] == 3#"Tdist"
            y = 2.1 .+ 0.5 .* x + rand(TDist(5), n)
            ε = [0.6, 0.25]
        else #"Chi"
            y = 2.1 .+ 0.5 .* x + rand(Chi(3), n)
            ε = [0.8, 1.]
        end
        # bayesian
        par = Sampler(y, X, 0.5, 10000, 5, 2500)
        β, θ, σ, α = mcmc(par, 0.8, .25, 1.5, 1, 2, 0.5, [2.1, 0.5])
        μ = X * median(β, dims = 1)' |> x -> reshape(x, size(x, 1))

        # Freq
        control[:est_sigma], control[:est_tau], control[:est_p] = (true, true, true)
        res = quantfreq(y, X, control)
        μf = X * res[:beta]

        b = DataFrame(hcat(par.y, par.X), :auto) |> x ->
            qreg(@formula(x1 ~  x3), x, settings[i, 1]) |> coef;
        q = X * b;
        #τ = [quantconvert(q[k], median(θ), median(α), μ[k], median(σ)) for k in 1:length(par.y)] |> mean

        b = DataFrame(hcat(par.y, par.X), :auto) |> x -> qreg(@formula(x1 ~  x3), x, settings[i, 1]) |> coef
        q = X * b
        if n >= 250
            taubayes = [quantconvert(q[k], median(θ), median(α), μ[k], median(σ)) for k in 1:length(y)] |> mean
            taufreq  = [quantconvert(q[k], res[:p], res[:tau], μf[k], res[:sigma]) for k in 1:length(y)] |> mean
        else
            taubayes = mcτ(α[i], median(α), median(θ), median(σ), 2500)
            taufreq  = mcτ(α[i], res[:tau], res[:p], res[:sigma], 2500)
        end

        par.α = taubayes
        βres= mcmc(par, 1.3, median(θ), median(σ), b);

        par.α = settings[i, 1]
        par.πθ = "uniform"
        βt, _, _ = mcmc(par, ε[1], ε[2], 1.5, 2, b)

        control[:est_sigma], control[:est_tau], control[:est_p] = (false, false, false)
        res = quantfreq(y, X, control, res[:sigma], res[:p], taufreq)

        freq[j] = mean(y .<= X*res[:beta])
        bayes[j] = [par.y[k] <= X[k,:] ⋅ median(βres, dims = 1)  for k in 1:length(par.y)] |> mean
        old[j] = [par.y[k] <= X[k,:] ⋅ median(βt, dims = 1)  for k in 1:length(par.y)] |> mean
        end
    settings[i, 3] = mean(bayes)
    settings[i, 4] = √var(bayes)
    settings[i, 5] = mean(freq)
    settings[i, 6] = √var(freq)
    settings[i, 7] = mean(old)
    settings[i, 8] = √var(old)
end

plt_dat = DataFrame(Tables.table(settings)) |> x -> rename!(x, cols)
CSV.write("C:/Users/lukar818/Dropbox/PhD/research/applied/quantile/R/plots/simulations/simsother250.csv", plt_dat)

## Bootstrap τ on davids data?
reps = 20
res = SharedArray{Float64}(reps)
quant = 0.9
@sync @distributed for i ∈ 1:reps
    println(i)
    ids = sample(1:length(y), length(y); replace = true)
    par = Sampler(y[ids], X[ids,:], 0.5, 10000, 5, 2000)
    b = DataFrame(hcat(par.y, par.X), :auto) |> x ->
        qreg(@formula(x1 ~  x3 + x4 + x5 + x6 + x7 + x8 + x9 + x10), x, 0.5) |> coef
    β, θ, σ, α = mcmc(par, 1.2, .35, 1.5, 1, 2, 0.5, b)

    b = DataFrame(hcat(par.y, par.X), :auto) |> x ->
        qreg(@formula(x1 ~  x3 + x4 + x5 + x6 + x7 + x8 + x9 + x10), x, quant) |> coef;
    q = par.X * b;
    μ = par.X * median(β, dims = 1)' |> x -> reshape(x, size(x, 1))
    res[i] = [quantconvert(q[j], median(θ), median(α), μ[j],
        median(σ)) for j in 1:length(par.y)] |> mean
end

τ = mean(res)
√var(res)

## Quantile with misspecified τ
n = 1000;
x = rand(Normal(), n);
X = hcat(ones(n), x)

α = range(0.1, 0.9, length = 17) |> Vector
res3 = SharedArray{Float64}(length(α))
reps = 50
@sync @distributed for i ∈ 1:length(α)
    temp = 0
    for j in 1:reps
        y = 2.1 .+ 0.5 .* x + rand(Aepd(0, 1, 0.7, 0.7), n);
        par = Sampler(y, X, α[i], 5000, 4, 1000)
        b = DataFrame(hcat(par.y, par.X), :auto) |> x ->
            qreg(@formula(x1 ~ x3), x, α[i]) |> coef
        β, _, _ = mcmc(par, 1., 1., 1.2, 4, b)
        temp += abs(α[i] - ([par.y[i] <= X[i,:] ⋅ median(β, dims = 1)  for i in 1:length(par.y)] |> mean))/reps
    end
    res3[i] = temp
end

vcat(hcat(res, α, [2. for i in 1:17]), hcat(res2, α, [1.5 for i in 1:17]),
    hcat(res3, α, [1. for i in 1:17])) |> x ->
    DataFrame(x, ["diff", "quant", "p"]) #|> x -> CSV.write("quantest.csv", x)


plot!(α, res3)

##
α = range(0.01, 0.99, length = 101);
p = [0.75, 1.5, 2, 3];
ε = [0.05, 0.2, 0.1, 0.05];
N = 20
n = 2000
τ = SharedArray{Float64}(length(α), length(p), N)

@sync @distributed for j ∈ 1:length(p)
    for iter ∈ 1:N
        y = 0.5 .+ rand(Normal(), n);
        inits = DataFrame(hcat(y, ones(n)), :auto) |> x -> qreg(@formula(x1 ~  1), x, 0.5) |> coef
        par = Sampler(y, hcat(ones(n)), 0.5, 16000, 5, 6000);
        β, σ = mcmc(par, ε[j], p[j], inits, 1., verbose = false)
        β = median(β[:,1])
        σ = median(σ)

        for i ∈ 1:length(α)
            q = (DataFrame(hcat(y, ones(n)), :auto) |> x -> qreg(@formula(x1 ~  1), x, α[i]) |> coef)[1]
            τ[i, j, iter] = quantconvert(q, p[j], 0.5, β, σ)
        end
    end
end


tau = mean(τ, dims = 3)[:,:,1]

plot(α, α)
plot!(α, tau[:,4])

DataFrame(tau = reshape(tau, (404,)),
    p = repeat(p, inner = 101),
    a = repeat(α, outer = 4)) |> x -> save("res.csv", x)

##
dat = load(string(pwd(), "/Tests/data/hks_jvdr.csv")) |> DataFrame;
y = dat[:, :osvAll]
X = dat[:, Not(["osvAll"])] |> Matrix
X = X[y.>0,:];
y = y[y.>0];
X = hcat([1 for i in 1:length(y)], X);

colnames = ["intercept" ; names(dat[:, Not(["osvAll"])])]
βinit = [-0.48, -0.14, -2.6, 3.7, 0., 0.1, 1.75, -0.05, 0.28]
M = 40
p = 0.05
feMCMC = FormatExpr("mcmc_{}.csv")
α = range(0.1, 0.9, length = 9)
ϵ = [0.9, 1.2, 1.3, 1.5, 1.5, 1.6, 1.6, 1.4, 0.6]

for i ∈ 1:length(α)
    par = Sampler(y, X, α[i], 150000, 30, 30000);
    N = Integer((par.nMCMC - par.burnIn)/par.thin) + 1
    β = SharedArray{Float64}((N, size(X)[2], M))
    σ = SharedArray{Float64}((N,M))
    θ = SharedArray{Float64}((N,M))

    @sync @distributed for j ∈ 1:M
        βt,θt,σt = mcmc(par, 10000., 1., ϵ[i], βinit, 1., 1.)
        β[:,:,j] = βt
        θ[:,j] = θt
        σ[:,j] = σt
    end

    a = DataFrame(param = colnames,
        value =[mean([median(β[:, j, i]) for i in 1:M]) for j in 1:size(X)[2]],
        l = [mean([sort(β, dims = 1)[Integer(round((p/2) * N)), j, i] for i in 1:M]) for j in 1:size(X)[2]],
        u =  [mean([sort(β, dims = 1)[Integer(round((1-p/2) * N)), j, i] for i in 1:M]) for j in 1:size(X)[2]], α = α[i])

    b = DataFrame(param = ["sigma", "theta"], value = [mean([median(σ[:, i]) for i in 1:M]), mean([median(θ[:, i]) for i in 1:M])],
            l = [mean(sort(σ, dims = 1)[Integer(round((p/2) * N)), :]) ; mean(sort(θ, dims = 1)[Integer(round((p/2) * N)), :])],
            u = [mean(sort(σ, dims = 1)[Integer(round((1-p/2) * N)), :]); mean(sort(θ, dims = 1)[Integer(round((1-p/2) * N)), :])], α = α[i])

    CSV.write(format(feMCMC, α[i]), vcat(a, b))
end

par = Sampler(y, X, 0.5, 150000, 30, 30000);
N = Integer((par.nMCMC - par.burnIn)/par.thin) + 1
M = 10
β = SharedArray{Float64}((N, size(X)[2], M))
σ = SharedArray{Float64}((N,M))
θ = SharedArray{Float64}((N,M))
@sync @distributed for i = 1:M
    βt,θt,σt =  mcmc(par, 10000., 1., 1.2, βinit, 1., 1.)
    β[:,:,i] = βt
    θ[:,i] = θt
    σ[:,i] = σt
end

a = DataFrame(param = colnames,
    value =[mean([median(β[:, j, i]) for i in 1:M]) for j in 1:size(X)[2]],
    l = [mean([sort(β, dims = 1)[Integer(round((p/2) * N)), j, i] for i in 1:M]) for j in 1:size(X)[2]],
    u =  [mean([sort(β, dims = 1)[Integer(round((1-p/2) * N)), j, i] for i in 1:M]) for j in 1:size(X)[2]], α = 0.5)

b = DataFrame(param = ["sigma", "theta"], value = [mean([median(σ[:, i]) for i in  1:M]), mean([median(θ[:, i]) for i in 1:M])],
        l = [mean(sort(σ, dims = 1)[Integer(round((p/2) * N)), :]) ; mean(sort(θ, dims = 1)[Integer(round((p/2) * N)), :])],
        u = [mean(sort(σ, dims = 1)[Integer(round((1-p/2) * N)), :]); mean(sort(θ, dims = 1)[Integer(round((1-p/2) * N)), :])], α = 0.5)

CSV.write(format(feMCMC, 0.5), vcat(a, b))
mean(σ)

(1:M)[Not(5)]

[median(σ[:, i]) for i in 1:M]
[median(θ[:, i]) for i in 1:M]

print(a)
print(b)


## Prediction
α = 0.8
reps = 100
aepd = SharedVector{Float64}(reps)
freq = SharedVector{Float64}(reps)

@sync @distributed for j ∈ 1:reps
    ids = sample(1:length(y), length(y)-100; replace = false)
    trainX, trainy = X[ids,:], y[ids]
    testX, testy = X[Not(ids),:], y[Not(ids)]
    par = Sampler(trainy, trainX, α, 21000, 5, 1000)

    βinit = DataFrame(hcat(par.y, par.X), :auto) |> x ->
        qreg(@formula(x1 ~  x3 + x4 + x5 + x6 + x7 + x8 + x9 + x10), x, par.α) |> coef
    β, θ, σ = mcmc(par, 0.4, 0.4, 4., 1.4, βinit);
    βest = [mean(β[:,i]) for i in 1:9]

    Q = zeros(100)
    Q2 = zeros(100)
    for i in 1:100
        Q[i] = testy[i] <= ceil(exp(testX[i,:] ⋅ βest) + par.α - 1)
        Q2[i] = testy[i] <= ceil(exp(testX[i,:] ⋅ βinit) + par.α - 1)
    end
    aepd[j] = mean(Q)
    freq[j] = mean(Q2)
end

mean(aepd[aepd.>0])
mean(freq[aepd.>0])

print(aepd)

√var(aepd[aepd.>0])
√var(freq[aepd.>0])
print(freq)
