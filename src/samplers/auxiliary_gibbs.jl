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
    ρ_block(l) = !own && l == L ? ρ_O : map(d -> d[blocks[l]], dicts)

    lab = zeros(Int, n)                 # 0 for j ∈ M
    for l in 1:L, j in blocks[l]
        lab[j] = l
    end
    w = _link_weights(log_DDCRP, i)     # w[j]: link weight of i → j

    # log p(y_M | ρ_B), NaN until computed
    logw = fill(NaN, L)
    lik(l) = isnan(logw[l]) ? (logw[l] = moving_set_loglik(model, M, ρ_block(l), blocks[l], state, data, priors)) : logw[l]

    # Far blocks (link weight below exp(-skip) of the largest) leave the Gibbs draw and are
    # reached by an MH move between them and the near blocks
    near = trues(L)
    if isfinite(p.skip)
        W = zeros(L + 1)                # L + 1: M
        @inbounds for j in 1:n
            W[lab[j] == 0 ? L + 1 : lab[j]] += w[j]
        end
        ref = log(maximum(W))
        for l in 1:L
            near[l] = log(W[l]) >= ref - p.skip
        end
        if !own && !all(near)
            done, j_new = _cross_far!(model, i, state, M, R, table_O, ρ_O, blocks, lab, near, W, w, lik)
            done && return (:gibbs, j_new, true)
            near[L] || return (:gibbs, state.c[i], true)
        end
    end
    for l in 1:L
        near[l] ? lik(l) : (logw[l] = -Inf)
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

    # p(c_i = j) ∝ w[j] × (likelihood factor of j's block), one exp per block
    mx = max(log_aux, maximum(logw; init = -Inf))
    f = exp.(logw .- mx)
    fM = exp(log_aux - mx)
    pj = Vector{Float64}(undef, n)
    @inbounds for j in 1:n
        pj[j] = w[j] * (lab[j] == 0 ? fM : f[lab[j]])
    end
    j_new = sample(1:n, Weights(pj))
    l = lab[j_new]

    if l == 0
        h = sample(1:m, Weights(exp.(a .- maximum(a))))
        for k in keys(dicts)
            delete!(dicts[k], table_O)
            dicts[k][M] = aux[h][k]
            own || (dicts[k][R] = ρ_O[k])
        end
    elseif !(!own && l == L)
        _join_block!(dicts, table_O, M, R, ρ_O, blocks[l])
    end                                 # j ∈ R: unchanged
    state.c[i] = j_new

    return (:gibbs, j_new, true)
end

_link_weights(L::DDCRPMatrix, i) = view(L.wt, :, i)
_link_weights(L::AbstractMatrix, i) = exp.(view(L, i, :))

# M leaves its cluster O (R keeps ρ_O if non-empty) and joins block B with B's parameters
function _join_block!(dicts, table_O, M, R, ρ_O, B)
    MB = sorted_merge(M, B)
    ρ_B = map(d -> d[B], dicts)
    for k in keys(dicts)
        delete!(dicts[k], table_O)
        delete!(dicts[k], B)
        dicts[k][MB] = ρ_B[k]
        isempty(R) || (dicts[k][R] = ρ_O[k])
    end
end

# MH move of c_i between the near and far existing blocks of c_{-i} (current link in R, the last
# block): j' is proposed from the link weights on the other side, so the ratio is
# p(y_M | ρ_B') / p(y_M | ρ_O) × W(proposed side) / W(current side). Returns (accepted, c_i).
function _cross_far!(model, i, state, M, R, table_O, ρ_O, blocks, lab, near, W, w, lik)
    L = length(blocks)
    cur_far = !near[L]
    W_near = sum(W[l] for l in 1:L if near[l]; init = 0.0)
    W_far = sum(W[l] for l in 1:L if !near[l]; init = 0.0)
    W_to = cur_far ? W_near : W_far
    W_to > 0 || return false, state.c[i]
    r = rand() * W_to
    j_new = 0
    @inbounds for j in eachindex(lab)
        (lab[j] != 0 && near[lab[j]] == cur_far) || continue
        j_new = j
        (r -= w[j]) <= 0 && break
    end
    l = lab[j_new]
    lr = lik(l) - lik(L) + log(W_to) - log(cur_far ? W_far : W_near)
    log(rand()) < lr || return false, state.c[i]
    _join_block!(cluster_param_dicts(state), table_O, M, R, ρ_O, blocks[l])
    state.c[i] = j_new
    return true, j_new
end
