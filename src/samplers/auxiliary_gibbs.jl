# ============================================================================
# Auxiliary-variable Gibbs update for customer assignments (Neal's Algorithm 8)
# ============================================================================

using StatsBase

sample_birth_params(model::LikelihoodModel, p::AuxiliaryGibbs, S_i::Vector{Int},
                    state::AbstractMCMCState, data::AbstractObservedData, priors::AbstractPriors) =
    sample_birth_params(model, p.proposal, S_i, state, data, priors)

birth_params_logpdf(model::LikelihoodModel, p::AuxiliaryGibbs, params_old::NamedTuple,
                    S_i::Vector{Int}, state::AbstractMCMCState, data::AbstractObservedData,
                    priors::AbstractPriors) =
    birth_params_logpdf(model, p.proposal, params_old, S_i, state, data, priors)

"""
    contribution_with(model, table, params, state, data, priors)

`table_contribution` of `table` with cluster parameters `params`, inserting the
key into the parameter dicts for the evaluation and restoring them afterwards.
"""
function contribution_with(model::LikelihoodModel, table::Vector{Int}, params::NamedTuple,
                           state::AbstractMCMCState, data::AbstractObservedData,
                           priors::AbstractPriors)
    dicts = cluster_param_dicts(state)
    had = haskey(first(dicts), table)
    saved = had ? map(d -> d[table], dicts) : nothing
    for k in keys(dicts)
        dicts[k][table] = params[k]
    end
    lc = table_contribution(model, table, state, data, priors)
    for k in keys(dicts)
        had ? (dicts[k][table] = saved[k]) : delete!(dicts[k], table)
    end
    return lc
end

"""
    cluster_logprior(model, params, priors)

Log prior density of one cluster's parameters, or `nothing` (the default). A
model that defines it declares that its `table_contribution` is
`Σ_{i ∈ table} log f(y_i | params) + cluster_logprior(model, params, priors)`,
which lets the auxiliary samplers score a moving set under a cluster's
parameters in O(|M|) instead of through contributions of the whole cluster.
"""
cluster_logprior(::LikelihoodModel, params, priors) = nothing

"""
    obs_loglik(model, i, params, data)

`log f(y_i | params)` for one observation, or `nothing` (the default). Optional:
a fast path for proposal inputs that score observations one at a time (the
two-anchor pilot); without it they use singleton `table_contribution`s.
"""
obs_loglik(::LikelihoodModel, i, params, data) = nothing

# log p(y_M | ρ) for the moving set M joining block B, whose parameters are ρ
function moving_set_loglik(model::LikelihoodModel, M::Vector{Int}, ρ::NamedTuple, B::Vector{Int},
                           state::AbstractMCMCState, data::AbstractObservedData, priors::AbstractPriors)
    lp = cluster_logprior(model, ρ, priors)
    isnothing(lp) || return contribution_with(model, M, ρ, state, data, priors) - lp
    return contribution_with(model, sorted_merge(M, B), ρ, state, data, priors) -
           contribution_with(model, B, ρ, state, data, priors)
end

function _logsumexp(x::AbstractVector{Float64})
    mx = maximum(x)
    isfinite(mx) || return mx
    return mx + log(sum(exp(v - mx) for v in x))
end

function update_c_i!(
    model::LikelihoodModel,
    i::Int,
    state::AbstractMCMCState,
    data::AbstractObservedData,
    priors::AbstractPriors,
    log_DDCRP::AbstractMatrix,
    p::AuxiliaryGibbs,
    ::FixedDimensionProposal,
    ::LinkProposal,
    ::StandardUpdate
)
    n = length(state.c)
    dicts = cluster_param_dicts(state)
    primary = first(dicts)

    table_O = find_table_for_customer(i, primary)
    M = get_moving_set(i, state.c, table_O)
    R = sorted_setdiff(table_O, M)
    own = isempty(R)
    ρ_O = map(d -> d[table_O], dicts)

    # Blocks other than M; R, if non-empty, is last and carries ρ_O
    blocks = Vector{Vector{Int}}()
    for B in keys(primary)
        B == table_O || push!(blocks, B)
    end
    own || push!(blocks, R)
    L = length(blocks)

    # log p(y_M | ρ_B)
    logw = Vector{Float64}(undef, L)
    for l in 1:L
        B = blocks[l]
        ρ_B = !own && l == L ? ρ_O : map(d -> d[B], dicts)
        logw[l] = moving_set_loglik(model, M, ρ_B, B, state, data, priors)
    end

    # Auxiliaries; the current ρ_M takes a random slot when M is on its own
    m = p.m
    aux = Vector{typeof(ρ_O)}(undef, m)
    a = Vector{Float64}(undef, m)
    h_cur = own ? rand(1:m) : 0
    for t in 1:m
        if t == h_cur
            aux[t] = ρ_O
            log_q = birth_params_logpdf(model, p.proposal, ρ_O, M, state, data, priors)
        else
            aux[t], log_q = sample_birth_params(model, p.proposal, M, state, data, priors)
        end
        a[t] = contribution_with(model, M, aux[t], state, data, priors) - log_q
        isnan(a[t]) && (a[t] = -Inf)
    end
    log_aux = _logsumexp(a) - log(m)

    lab = zeros(Int, n)                 # 0 for j ∈ M
    for l in 1:L, j in blocks[l]
        lab[j] = l
    end
    logp = Vector{Float64}(undef, n)
    @inbounds for j in 1:n
        logp[j] = log_DDCRP[i, j] + (lab[j] == 0 ? log_aux : logw[lab[j]])
    end
    mx = maximum(logp)
    j_new = sample(1:n, Weights(exp.(logp .- mx)))
    l = lab[j_new]

    if l == 0
        h = sample(1:m, Weights(exp.(a .- maximum(a))))
        for k in keys(dicts)
            delete!(dicts[k], table_O)
            dicts[k][M] = aux[h][k]
            own || (dicts[k][R] = ρ_O[k])
        end
    elseif !(!own && l == L)
        B = blocks[l]
        MB = sorted_merge(M, B)
        ρ_B = map(d -> d[B], dicts)
        for k in keys(dicts)
            delete!(dicts[k], table_O)
            delete!(dicts[k], B)
            dicts[k][MB] = ρ_B[k]
            own || (dicts[k][R] = ρ_O[k])
        end
    end                                 # j ∈ R: unchanged
    state.c[i] = j_new

    return (:gibbs, j_new, true)
end
