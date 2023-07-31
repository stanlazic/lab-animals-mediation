using Pkg
Pkg.activate(".")

# run once to install packages
# Pkg.instantiate()

using Statistics
using Plots
using StatsPlots
using LaTeXStrings
using Distributions
using Turing
using Random
using DataFrames
using GLM


# ------------------------------------------------------------
# Define Bayesian model
# ------------------------------------------------------------


@model function m1(t, m, o)

    # priors
    a1 ~ Normal(0, 10) 
    a2 ~ Normal(0, 10)
    B1 ~ Normal(0, 10)
    B2 ~ Normal(0, 10)
    B3 ~ Normal(0, 10)

    σm ~ truncated(Normal(0, 10), 0, Inf)
    σt ~ truncated(Normal(0, 10), 0, Inf)
    
    for i in 1:length(o)
        # effect of T on M (B1)
        m[i] ~ Normal(a1 + B1 * t[i], σm)
        
        # effect of M on O adjusted for T (B3) and effect of T on O adjusted for M (B2)
        o[i] ~ Normal(a2 + B2 * t[i] + B3 * m[i], σt)
    end
end


# set sample size per group
N = 12


# ------------------------------------------------------------
# Mediation model (indirect effect only)
# ------------------------------------------------------------

# DAG
p1 = plot(xlim=(0, 1), ylim=(0, 1), showaxis=false, grid=false)
annotate!(0.25, 0.25, text(L"T", 20))
annotate!(0.5, 0.5, text(L"M", 20))
annotate!(0.75, 0.25, text(L"O", 20))
plot!([0.29, 0.46], [0.29, 0.46], arrow=true, color=:steelblue, linewidth=2, label=:none)
plot!([0.54, 0.71], [0.46, 0.29], arrow=true, color=:steelblue, linewidth=2, label=:none)


# coefficients
Random.seed!(123)
βtm = 2.5
βmo = 2.5
βto = 0
σ = 2


# generate data
T1 = repeat([0, 1], N)
M1 = 3 .+ βtm .* T1 .+ rand(Normal(0, σ), N * 2)
O1 = 2 .+ βmo .* M1 .+ βto .* T1.+ rand(Normal(0, σ), N * 2)

# test treatment effect on mediator and outcome
lm(@formula(M1 ~ T1), DataFrame(M1=M1, T1=T1))
lm(@formula(O1 ~ T1), DataFrame(O1=O1, T1=T1))



p4 = bar([mean(M1[T1 .== 0]), mean(M1[T1 .== 1])], label=:none, bar_width=0.6, 
         xlim=(0.25, 2.75), xticks=([1, 2], ["Control", "Treated"]), 
         yerr=[std(M1[T1 .== 0]) / sqrt(N), std(M1[T1 .== 1]) / sqrt(N)], ylabel="Mediator\n",
         markerstrokecolor=:black, linecolor=:black, ylim=(0, 6.5), 
         fillcolour=:black, fillalpha=0.4, widen=false)

p7 = bar([mean(O1[T1 .== 0]), mean(O1[T1 .== 1])], label=:none, bar_width=0.6, 
         xlim=(0.25, 2.75), xticks=([1, 2], ["Control", "Treated"]), 
         yerr=[std(O1[T1 .== 0]) / sqrt(N), std(O1[T1 .== 1]) / sqrt(N)], ylabel="Outcome\n",
         markerstrokecolor=:black, linecolor=:black, ylim=(0, 20),
         fillcolour=:black, fillalpha=0.4, widen=false)


# fit Bayesian model
post = sample(m1(T1, M1, O1), NUTS(), MCMCThreads(), 5_000, 3)

df_post = DataFrame(post)

# calculate mediated effect 
df_post.med_eff = df_post.B1 .* df_post.B3


p10 = density(df_post.B2, label="Direct", trim=true,
             colour=:firebrick, fill=true, fillalpha=0.4,
             xlab="Effect on outcome", xlim=(-5, 12), ylim=(0, 0.8))
density!(df_post.med_eff, label="Mediated", trim=true,
         colour=:steelblue, fill=true, fillalpha=0.4)
vline!([0], color=:black, linestyle=:dash, label=:none)




# ------------------------------------------------------------
# Confounding model (direct effect only)
# ------------------------------------------------------------

# DAG
p2 = plot(xlim=(0, 1), ylim=(0, 1), showaxis=false, grid=false)
annotate!(0.25, 0.25, text(L"T", 20))
annotate!(0.5, 0.5, text(L"M", 20))
annotate!(0.75, 0.25, text(L"O", 20))
plot!([0.29, 0.46], [0.29, 0.46], arrow=true, color=:steelblue, linewidth=2, label=:none)
plot!([0.29, 0.71], [0.25, 0.25], arrow=true, color=:firebrick, linewidth=2, label=:none)


# coefficients
Random.seed!(123)
βtm = 2.5
βmo = 0
βto = 5.5
σ = 2

# generate data
T2 = repeat([0, 1], N)
M2 = 3 .+ βtm .* T2 .+ rand(Normal(0, σ), N * 2)
O2 = 10 .+ βmo .* M2 .+ βto .* T2 .+ rand(Normal(0, σ * 1.75), N * 2)

# test treatment effect on mediator and outcome
lm(@formula(M2 ~ T2), DataFrame(M2=M2, T2=T2))
lm(@formula(O2 ~ T2), DataFrame(O2=O2, T2=T2))


p5 = bar([mean(M2[T2 .== 0]), mean(M2[T2 .== 1])], label=:none, bar_width=0.6, 
         xlim=(0.25, 2.75), xticks=([1, 2], ["Control", "Treated"]), 
         yerr=[std(M2[T2 .== 0]) / sqrt(N), std(M2[T2 .== 1]) / sqrt(N)], ylabel="Mediator\n",
         markerstrokecolor=:black, linecolor=:black, ylim=(0, 6.5),
         fillcolour=:black, fillalpha=0.4, widen=false)

p8 = bar([mean(O2[T2 .== 0]), mean(O2[T2 .== 1])], label=:none, bar_width=0.6, 
         xlim=(0.25, 2.75), xticks=([1, 2], ["Control", "Treated"]), 
         yerr=[std(O2[T2 .== 0]) / sqrt(N), std(O2[T2 .== 1]) / sqrt(N)], ylabel="Outcome\n",
         markerstrokecolor=:black, linecolor=:black, ylim=(0, 20),
         fillcolour=:black, fillalpha=0.4, widen=false)


# fit Bayesian model
post = sample(m1(T2, M2, O2), NUTS(), MCMCThreads(), 5_000, 3)

df_post = DataFrame(post)

# calculate mediated effect 
df_post.med_eff = df_post.B1 .* df_post.B3

# probability of mediated and direct effect
mean(df_post.med_eff .> 0)
mean(df_post.B2 .> 0)


p11 = density(df_post.B2, label="Direct", trim=true,
             colour=:firebrick, fill=true, fillalpha=0.4,
             xlab="Effect on outcome", xlim=(-5, 12), ylim=(0, 0.8))
density!(df_post.med_eff, label="Mediated", trim=true,
         colour=:steelblue, fill=true, fillalpha=0.4)
vline!([0], color=:black, linestyle=:dash, label=:none)



# ------------------------------------------------------------
# Two-effects  model (both effects)
# ------------------------------------------------------------

# DAG
p3 = plot(xlim=(0, 1), ylim=(0, 1), showaxis=false, grid=false)
annotate!(0.25, 0.25, text(L"T", 20))
annotate!(0.5, 0.5, text(L"M", 20))
annotate!(0.75, 0.25, text(L"O", 20))
plot!([0.29, 0.46], [0.29, 0.46], arrow=true, color=:steelblue, linewidth=2, label=:none)
plot!([0.54, 0.71], [0.46, 0.29], arrow=true, color=:steelblue, linewidth=2, label=:none)
plot!([0.29, 0.71], [0.25, 0.25], arrow=true, color=:firebrick, linewidth=2, label=:none)


# coefficients
Random.seed!(123)
βtm = 2
βmo = 2
βto = 2.5
σ = 2

# generate data
T3 = repeat([0, 1], N)
M3 = 3 .+ βtm .* T3 .+ rand(Normal(0, σ), N * 2)
O3 = 3 .+ βmo .* M3 .+ βto .* T3.+ rand(Normal(0, σ * 0.95), N * 2)

# test treatment effect on mediator and outcome
lm(@formula(M3 ~ T3), DataFrame(M3=M3, T3=T3))
lm(@formula(O3 ~ T3), DataFrame(O3=O3, T3=T3))


p6 = bar([mean(M3[T3 .== 0]), mean(M3[T3 .== 1])], label=:none, bar_width=0.6, 
         xlim=(0.25, 2.75), xticks=([1, 2], ["Control", "Treated"]), 
         yerr=[std(M3[T3 .== 0]) / sqrt(N), std(M3[T3 .== 1]) / sqrt(N)], ylabel="Mediator\n",
         markerstrokecolor=:black, linecolor=:black, ylim=(0, 6.5),
         fillcolour=:black, fillalpha=0.4, widen=false)

p9 = bar([mean(O3[T3 .== 0]), mean(O3[T3 .== 1])], label=:none, bar_width=0.6, 
         xlim=(0.25, 2.75), xticks=([1, 2], ["Control", "Treated"]), 
         yerr=[std(O3[T3 .== 0]) / sqrt(N), std(O3[T3 .== 1]) / sqrt(N)], ylabel="Outcome\n",
         markerstrokecolor=:black, linecolor=:black, ylim=(0, 20),
         fillcolour=:black, fillalpha=0.4, widen=false)


# fit Bayesian model
post = sample(m1(T3, M3, O3), NUTS(), MCMCThreads(), 5_000, 3)

df_post = DataFrame(post)

# calculate mediated effect 
df_post.med_eff = df_post.B1 .* df_post.B3

# probability of mediated and direct effect
mean(df_post.med_eff .> 0)
mean(df_post.B2 .> 0)


p12 = density(df_post.B2, label="Direct", trim=true,
             colour=:firebrick, fill=true, fillalpha=0.4,
             xlab="Effect on outcome", xlim=(-5, 12), ylim=(0, 0.8))
density!(df_post.med_eff, label="Mediated", trim=true,
         colour=:steelblue, fill=true, fillalpha=0.4)
vline!([0], color=:black, linestyle=:dash, label=:none)


# create figure and save
savefig(
    Plots.plot(p1, p2, p3,
           p4, p5, p6,
           p7, p8, p9,
           p10, p11, p12, 
           layout=(4, 3),
           tick_direction=:out,
           tickfontsize=12, guidefontsize=12, 
           size=(width=900, height=1000)),
    "fig1.svg"
)
