# ============================================================================
# Metropolised Algorithm 8: reversible jump link updates with an auxiliary pool
# ============================================================================

sample_birth_params(model::LikelihoodModel, p::AuxiliaryMH, S_i::Vector{Int},
                    state::AbstractMCMCState, data::AbstractObservedData, priors::AbstractPriors) =
    sample_birth_params(model, p.proposal, S_i, state, data, priors)

birth_params_logpdf(model::LikelihoodModel, p::AuxiliaryMH, params_old::NamedTuple,
                    S_i::Vector{Int}, state::AbstractMCMCState, data::AbstractObservedData,
                    priors::AbstractPriors) =
    birth_params_logpdf(model, p.proposal, params_old, S_i, state, data, priors)

# Pool of m parameters for M from q, with ρ_M in a random slot when M is on its own
# (`own`). Returns the pool, the log weights π·lik/q and the log of their mean.
function _auxiliary_pool(model, q, m, M, ρ_M::T, own, state, data, priors) where {T}
    aux = Vector{T}(undef, m)
    a = Vector{Float64}(undef, m)
    h_cur = own ? rand(1:m) : 0
    for t in 1:m
        if t == h_cur
            aux[t] = ρ_M
            log_q = birth_params_logpdf(model, q, ρ_M, M, state, data, priors)
        else
            aux[t], log_q = sample_birth_params(model, q, M, state, data, priors)
        end
        a[t] = contribution_with(model, M, aux[t], state, data, priors) - log_q
        isnan(a[t]) && (a[t] = -Inf)
    end
    return aux, a, _logsumexp(a) - log(m)
end

# Σ_{j ∉ b} w[j] for sorted b
function _mass_outside(w::AbstractVector{Float64}, b::Vector{Int})
    s = 0.0; k = 1
    @inbounds for j in eachindex(w)
        if k <= length(b) && b[k] == j
            k += 1
        else
            s += w[j]
        end
    end
    return s
end

# Draw j ∉ b (sorted) with probability ∝ w[j]; `out` is the mass outside b
function _sample_outside(w::AbstractVector{Float64}, b::Vector{Int}, out::Float64)
    u = rand() * out
    acc = 0.0; k = 1; j_last = 0
    @inbounds for j in eachindex(w)
        if k <= length(b) && b[k] == j
            k += 1
        else
            acc += w[j]; j_last = j
            acc >= u && return j
        end
    end
    return j_last                      # only reachable through rounding
end

# Draw j ∈ b with probability ∝ w[j]
function _sample_within(w::AbstractVector{Float64}, b::Vector{Int})
    u = rand() * sum(j -> w[j], b)
    acc = 0.0
    @inbounds for j in b
        acc += w[j]
        acc >= u && return j
    end
    return last(b)
end

function update_c_i!(
    model::LikelihoodModel,
    i::Int,
    state::AbstractMCMCState,
    data::AbstractObservedData,
    priors::AbstractPriors,
    log_DDCRP::AbstractMatrix,
    p::AuxiliaryMH,
    ::FixedDimensionProposal,
    link::LinkProposal,
    ::StandardUpdate
)
    dicts = cluster_param_dicts(state)
    primary = first(dicts)
    j_old = state.c[i]

    table_O = find_table_for_customer(i, primary)
    M = get_moving_set(i, state.c, table_O)
    R = sorted_setdiff(table_O, M)
    own = isempty(R)
    ρ_O = map(d -> d[table_O], dicts)
    b_old = own ? M : R                 # block of z(c_{-i}) holding j_old

    # Propose j*; `corr` is the log prior ratio times the proposal ratio
    metro = link isa MetropolisedPriorLink
    if metro
        log_DDCRP isa DDCRPMatrix || throw(ArgumentError("MetropolisedPriorLink needs a DDCRPMatrix"))
        w = view(log_DDCRP.wt, :, i)
        out_old = _mass_outside(w, b_old)
        if !(out_old > 0)
            state.c[i] = _sample_within(w, b_old)
            return (:fixed, state.c[i], true)
        end
        j_star = _sample_outside(w, b_old, out_old)
    else
        j_star, corr = propose_link(i, j_old, log_DDCRP, link)
    end

    star_M = insorted(j_star, M)
    star_R = !own && insorted(j_star, R)
    B = star_M ? M : star_R ? R : find_table_for_customer(j_star, primary)
    metro && (corr = log(out_old) - log(_mass_outside(w, B)))

    if (own && star_M) || star_R
        # Same block: the partition and the likelihood are unchanged
        accepted = log(rand()) < corr
        accepted && (state.c[i] = j_star)
        return (:fixed, j_star, accepted)
    end

    if !own && !star_M
        # M moves from O to B; both keep their parameters
        ρ_B = map(d -> d[B], dicts)
        log_α = moving_set_loglik(model, M, ρ_B, B, state, data, priors) -
                moving_set_loglik(model, M, ρ_O, R, state, data, priors) + corr
        move = :fixed
        accepted = log(rand()) < log_α
        if accepted
            MB = sorted_merge(M, B)
            for k in keys(dicts)
                delete!(dicts[k], table_O); delete!(dicts[k], B)
                dicts[k][R] = ρ_O[k]; dicts[k][MB] = ρ_B[k]
            end
        end
    elseif !own
        # Birth: M leaves O for a cluster of its own
        aux, a, log_W = _auxiliary_pool(model, p.proposal, p.m, M, ρ_O, false, state, data, priors)
        log_α = log_W - moving_set_loglik(model, M, ρ_O, R, state, data, priors) + corr
        move = :birth
        accepted = log(rand()) < log_α
        if accepted
            h = sample(1:p.m, Weights(exp.(a .- maximum(a))))
            for k in keys(dicts)
                delete!(dicts[k], table_O)
                dicts[k][R] = ρ_O[k]; dicts[k][M] = aux[h][k]
            end
        end
    else
        # Death: M's cluster joins B
        ρ_B = map(d -> d[B], dicts)
        aux, a, log_W = _auxiliary_pool(model, p.proposal, p.m, M, ρ_O, true, state, data, priors)
        log_α = moving_set_loglik(model, M, ρ_B, B, state, data, priors) - log_W + corr
        move = :death
        accepted = log(rand()) < log_α
        if accepted
            MB = sorted_merge(M, B)
            for k in keys(dicts)
                delete!(dicts[k], M); delete!(dicts[k], B)
                dicts[k][MB] = ρ_B[k]
            end
        else
            # Gibbs step on the pool slot given j ∈ M refreshes ρ_M
            h = sample(1:p.m, Weights(exp.(a .- maximum(a))))
            for k in keys(dicts)
                dicts[k][M] = aux[h][k]
            end
        end
    end
    accepted && (state.c[i] = j_star)

    # Exact redraw of c_i within its block (partition unchanged)
    metro && (state.c[i] = _sample_within(w, accepted ? B : b_old))
    return (move, j_star, accepted)
end
