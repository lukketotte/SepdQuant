include("QR.jl")
using .QR, Distributions, LinearAlgebra, StatsBase, SpecialFunctions
using Plots, PlotThemes, CSV, DataFrames, StatFiles
theme(:juno)


dat = load("C:\\Users\\lukar818\\Documents\\PhD\\SMC\\Tests\\data\\nsa_ff.dta") |> DataFrame
names(dat)
dropmissing(dat)
