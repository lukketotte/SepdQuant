using Plots, Formatting, SpecialFunctions, EpdTest, Distributions


function δ(p, α)
    2*α^p*(1-α)^p / (α^p + (1-α)^p)
end

function expTerm(x, α, μ, σ, θ)
    if x < μ
        (μ - x)^θ / (α*σ)^θ
    else
        (x-μ)^θ / ((1-α)*σ)^θ
    end
end

## Bivariate
f(x, y, p) = pdf(MvEpd([0., 0.], 1., p),[x' ; y'])
x = range(-4, 4, length = 1000)

surface(x, x, f)
surface(x, x, f.(x', x, 10.))
plot(x, x, f.(x', x, .1), st=:surface, zlims=(0., 0.3), title="test")



@userplot MvEpdPlot
@recipe function p(mvepd::MvEpdPlot)
    x, p = mvepd.args
    x, x, f.(x',x,p)
end

anim = @animate for θ ∈ range(0.1, 5, length = 100)
    mvepdplot(x, θ, st=:surface,
    title=string("β = ", round(θ, digits = 3)), zlims=(0., 0.3),
    c=:thermal, camera=(-30,30))
end
gif(anim, "anim_theta.gif", fps = 15)

## Univariate
@userplot AepdPlot
@recipe function p(ap::AepdPlot)
    x, μ, σ, θ, α = ap.args
    del = δ(θ, α)
    C = del^(1/θ) / (σ*gamma(1+1/θ))
    x, C .* exp.(-del .* expTerm.(x, α, μ, σ, θ))
end


x = range(-7, 7, length = 1000)


anim = @animate for θ ∈ range(0.1, 5, length = 100)
    aepdplot(x, 0., 1., θ, 0.3,
        label=string("θ= ", round(θ, digits = 3), " α = 0.3"),
        xtickfontsize=12,ytickfontsize=12,xguidefontsize=12,
        xlabel="x",legendfontsize=14, linewidth=2)
    aepdplot!(x, 0., 1., θ, 0.5,
        label=string("θ= ", round(θ, digits = 3), " α = 0.5"),
        xtickfontsize=12,ytickfontsize=12,xguidefontsize=12,
        xlabel="x",legendfontsize=14, linewidth=2)
    aepdplot!(x, 0., 1., θ, 0.9,
        label=string("θ= ", round(θ, digits = 3), " α = 0.9"),
        xtickfontsize=12,ytickfontsize=12,xguidefontsize=12,
        xlabel="x",legendfontsize=14, linewidth=2)
end
gif(anim, "anim_theta.gif", fps = 15)

x = range(-7, 7, length = 1000)
anim = @animate for α ∈ range(0.1, 0.9, length = 100)
    aepdplot(x, 0., 1., 1., α,
        label=string("α = ", round(α, digits = 2), " θ = 1."),
        xtickfontsize=12,ytickfontsize=12,xguidefontsize=12,
        xlabel="x",legendfontsize=14, linewidth=2)
    aepdplot!(x, 0., 1., 2., α,
        label=string("α = ", round(α, digits = 2), " θ = 2."),
        xtickfontsize=12,ytickfontsize=12,xguidefontsize=12,
        xlabel="x",legendfontsize=14, linewidth=2)
    aepdplot!(x, 0., 1., 5., α,
        label=string("α = ", round(α, digits = 2), " θ = 5."),
        xtickfontsize=12,ytickfontsize=12,xguidefontsize=12,
        xlabel="x",legendfontsize=12, linewidth=2)
end
gif(anim, "anim_alpha.gif", fps = 15)
