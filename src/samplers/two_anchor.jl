# ============================================================================
# Two-anchor multi-link split/merge Metropolis-Hastings
# ============================================================================

two_anchor_redraw_prob(ta::TwoAnchor, nU::Int) = min(0.5, ta.k / nU)

"""
    two_anchor_logQ(c_from, c_to, U, p, L)
    two_anchor_logQ(c_from, c_to, U, ta, P, split, L)

Log probability that the two-anchor proposal moves the links of `U` from
`c_from` to `c_to`: each `i ∈ U` keeps its link with probability `1 − p_i` or
redraws it from `g_i` on `U`, so an unchanged link has mass `1 − p_i + p_i g_i(c_i)`.
The first form is the basic move (`p_i = p`, `g_i` ∝ ddCRP weights). The second
uses the redraw probabilities and targets of `ta` in direction `split`, guided
by the pilot `P` of `two_anchor_pilot` (`nothing` for the basic move).
"""
function two_anchor_logQ(c_from::AbstractVector{Int}, c_to::AbstractVector{Int}, U::Vector{Int},
                         p::Float64, L::DDCRPMatrix)
    lq = 0.0
    for i in U
        w = view(L.wt, :, i)
        g = w[c_to[i]] / sum(j -> w[j], U)
        lq += log((c_to[i] == c_from[i] ? 1 - p : 0.0) + p * g)
    end
    return lq
end

function two_anchor_logQ(c_from::AbstractVector{Int}, c_to::AbstractVector{Int}, U::Vector{Int},
                         ta::TwoAnchor, P, split::Bool, L::DDCRPMatrix)
    lq = 0.0
    for i in U
        w = view(L.wt, :, i)
        g = w[c_to[i]] * _ta_tilt(ta, P, i, c_to[i], split) / sum(j -> w[j] * _ta_tilt(ta, P, i, j, split), U)
        p = _ta_redraw(ta, i, length(U), P)
        lq += log((c_to[i] == c_from[i] ? 1 - p : 0.0) + p * g)
    end
    return lq
end

# Redraw probability of i's link (data only, the same in both directions) and the target weight
# factor of j, given the pilot P
_ta_redraw(ta::TwoAnchor, i, nU, P) =
    (p0 = two_anchor_redraw_prob(ta, nU); isnothing(P) ? p0 : max(p0, ta.p_cross * P.share[i]))
_ta_tilt(ta::TwoAnchor, P, i, j, split) =
    (isnothing(P) || ta.tilt == 1 || (P.group[j] == P.group[i]) != split) ? 1.0 : ta.tilt

"""
    two_anchor_draw(c, U, ta, P, split, L) -> c_new

Draws the proposed links of `U` (see `two_anchor_logQ`).
"""
function two_anchor_draw(c::AbstractVector{Int}, U::Vector{Int}, ta::TwoAnchor, P, split::Bool, L::DDCRPMatrix)
    c_new = copy(c)
    tw = Vector{Float64}(undef, length(U))
    for i in U
        rand() < _ta_redraw(ta, i, length(U), P) || continue
        w = view(L.wt, :, i)
        for (k, j) in enumerate(U)
            tw[k] = w[j] * _ta_tilt(ta, P, i, j, split)
        end
        r = rand() * sum(tw); acc = 0.0; pick = U[end]
        for (k, j) in enumerate(U)
            acc += tw[k]
            if r <= acc; pick = j; break; end
        end
        c_new[i] = pick
    end
    return c_new
end

"""
    two_anchor_pilot(model, state, data, priors, L, ta, a, b) -> (group, share) or nothing

The pilot of `U` (the union of the anchors' clusters) when `ta.pilot`. `group[i]`
is `true` on `a`'s side: it starts from the nearer anchor by ddCRP weight, then
twice reassigns each `i` by `ℓ_i(ρ̂_a) − ℓ_i(ρ̂_b) + log w_ia − log w_ib`, where `ρ̂`
are `moment_fit`s of the two groups (prior mean when a fit is unavailable) and
`ℓ_i` is `i`'s singleton contribution. `share[i]` is the fraction of `i`'s link
weight to the rest of `U` that falls in the other group: links near the pilot
boundary are redrawn with probability `max(p, p_cross · share[i])`. The pilot
depends on `U`, the anchors and the data only, so the forward and reverse
proposals share it, and with `tilt = 1` the link terms of the ratio cancel.
"""
function two_anchor_pilot(model::LikelihoodModel, state::AbstractMCMCState, data::AbstractObservedData,
                          priors::AbstractPriors, L::DDCRPMatrix, ta::TwoAnchor, a::Int, b::Int)
    ta.pilot || return nothing
    primary = first(cluster_param_dicts(state))
    TA = find_table_for_customer(a, primary)
    TB = find_table_for_customer(b, primary)
    U = TA == TB ? TA : sorted_merge(TA, TB)
    g = [L[i, a] >= L[i, b] for i in U]
    g[findfirst(==(a), U)] = true; g[findfirst(==(b), U)] = false
    fit(S) = (f = moment_fit(model, S, data, priors); isnothing(f) ? prior_mean_params(model, priors) : f)
    function ll(i, ρ)
        v = obs_loglik(model, i, ρ, data)
        isnothing(v) || return v
        lp = cluster_logprior(model, ρ, priors)
        return contribution_with(model, [i], ρ, state, data, priors) - (isnothing(lp) ? 0.0 : lp)
    end
    for _ in 1:2
        ρa = fit(U[g])
        ρb = fit(U[.!g])
        for (k, i) in enumerate(U)
            (i == a || i == b) && continue
            g[k] = ll(i, ρa) - ll(i, ρb) + L[i, a] - L[i, b] >= 0
        end
    end
    s = zeros(length(U))
    for (k, i) in enumerate(U)
        w = view(L.wt, :, i)
        tot = 0.0; other = 0.0
        for (m, j) in enumerate(U)
            m == k && continue
            tot += w[j]
            g[m] != g[k] && (other += w[j])
        end
        s[k] = tot > 0 ? other / tot : 0.0
    end
    return (group = Dict(zip(U, g)), share = Dict(zip(U, s)))
end

"""
    two_anchor_components(c, U) -> (ncomp, label)

Weakly connected components of the links `i → c[i]`, `i ∈ U`, all inside `U`;
`label[i]` is the component of `i`.
"""
function two_anchor_components(c::AbstractVector{Int}, U::Vector{Int})
    parent = Dict(i => i for i in U)
    root(x) = (while parent[x] != x; parent[x] = parent[parent[x]]; x = parent[x]; end; x)
    for i in U
        ri, rj = root(i), root(c[i])
        ri != rj && (parent[ri] = rj)
    end
    label = Dict(i => root(i) for i in U)
    return length(unique(values(label))), label
end

"""
    two_anchor_logratio(model, state, data, priors, L, q, ta, a, b, c_new, ρ_B; P)

Log MH ratio of the two-anchor move from the current state to the links
`c_new` (valid topology assumed). On a split `ρ_B` is the new parameter of the
component holding `b`; on a merge it is ignored. `P` is the pilot
(computed when not given). Returns `(log ratio, split, A, B, U)`.
"""
function two_anchor_logratio(model::LikelihoodModel, state::AbstractMCMCState, data::AbstractObservedData,
                             priors::AbstractPriors, L::DDCRPMatrix, q::BirthProposal, ta::TwoAnchor,
                             a::Int, b::Int, c_new::AbstractVector{Int}, ρ_B;
                             P = two_anchor_pilot(model, state, data, priors, L, ta, a, b))
    dicts = cluster_param_dicts(state)
    primary = first(dicts)
    TA = find_table_for_customer(a, primary)
    TB = find_table_for_customer(b, primary)
    split = TA == TB
    U = split ? TA : sorted_merge(TA, TB)
    c = state.c

    lr = sum(L[i, c_new[i]] - L[i, c[i]] for i in U) +
         two_anchor_logQ(c_new, c, U, ta, P, !split, L) - two_anchor_logQ(c, c_new, U, ta, P, split, L)
    if split
        _, lab = two_anchor_components(c_new, U)
        A = [i for i in U if lab[i] == lab[a]]
        B = [i for i in U if lab[i] != lab[a]]
        ρ_A = map(d -> d[TA], dicts)
        lr += contribution_with(model, A, ρ_A, state, data, priors) +
              contribution_with(model, B, ρ_B, state, data, priors) -
              table_contribution(model, U, state, data, priors) -
              birth_params_logpdf(model, q, ρ_B, B, state, data, priors)
    else
        A, B = TA, TB
        ρ_A = map(d -> d[TA], dicts)
        lr += contribution_with(model, U, ρ_A, state, data, priors) -
              table_contribution(model, TA, state, data, priors) -
              table_contribution(model, TB, state, data, priors) +
              birth_params_logpdf(model, q, map(d -> d[TB], dicts), TB, state, data, priors)
    end
    return lr, split, A, B, U
end

"""
    two_anchor_update!(model, state, data, priors, L, q, ta, stats)

One two-anchor split/merge attempt (see `TwoAnchor`), recorded in `stats`.
"""
function two_anchor_update!(model::LikelihoodModel, state::AbstractMCMCState, data::AbstractObservedData,
                            priors::AbstractPriors, L::DDCRPMatrix, q::BirthProposal, ta::TwoAnchor,
                            stats::TwoAnchorStats)
    t0 = time()
    n = length(state.c)
    a = rand(1:n)
    b = rand(1:(n - 1)); b >= a && (b += 1)
    dicts = cluster_param_dicts(state)
    primary = first(dicts)
    TA = find_table_for_customer(a, primary)
    TB = find_table_for_customer(b, primary)
    split = TA == TB
    U = split ? TA : sorted_merge(TA, TB)
    stats.U_size_sum += length(U)
    split ? (stats.split_attempts += 1) : (stats.merge_attempts += 1)

    P = two_anchor_pilot(model, state, data, priors, L, ta, a, b)
    c_new = two_anchor_draw(state.c, U, ta, P, split, L)
    ncomp, lab = two_anchor_components(c_new, U)
    valid = split ? (ncomp == 2 && lab[a] != lab[b]) : ncomp == 1
    if !valid
        split ? (stats.split_invalid += 1) : (stats.merge_invalid += 1)
        stats.invalid_components[ncomp] = get(stats.invalid_components, ncomp, 0) + 1
        stats.time += time() - t0
        return false
    end
    nchg = count(i -> c_new[i] != state.c[i], U)
    push!(stats.valid_links_changed, nchg)

    ρ_B = nothing
    if split
        B = [i for i in U if lab[i] != lab[a]]
        ρ_B, _ = sample_birth_params(model, q, B, state, data, priors)
    end
    lr, _, A, B, _ = two_anchor_logratio(model, state, data, priors, L, q, ta, a, b, c_new, ρ_B; P = P)
    accepted = log(rand()) < lr
    if accepted
        ρ_A = map(d -> d[TA], dicts)
        for k in keys(dicts)
            delete!(dicts[k], TA); split || delete!(dicts[k], TB)
            if split
                dicts[k][A] = ρ_A[k]; dicts[k][B] = ρ_B[k]
            else
                dicts[k][U] = ρ_A[k]
            end
        end
        for i in U
            state.c[i] = c_new[i]
        end
        if split
            stats.split_accepts += 1; push!(stats.split_accepted_links, nchg)
        else
            stats.merge_accepts += 1; push!(stats.merge_accepted_links, nchg)
        end
    end
    stats.time += time() - t0
    return accepted
end
