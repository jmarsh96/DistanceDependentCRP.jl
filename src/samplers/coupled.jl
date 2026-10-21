# ============================================================================
# Coupled-parameter birth/death (shared by the reversible jump and RJ pool updates)
# ============================================================================

sample_birth_params(model::LikelihoodModel, p::CoupledRJ, S_i::Vector{Int},
                    state::AbstractMCMCState, data::AbstractObservedData, priors::AbstractPriors) =
    sample_birth_params(model, p.proposal, S_i, state, data, priors)

birth_params_logpdf(model::LikelihoodModel, p::CoupledRJ, params_old::NamedTuple,
                    S_i::Vector{Int}, state::AbstractMCMCState, data::AbstractObservedData,
                    priors::AbstractPriors) =
    birth_params_logpdf(model, p.proposal, params_old, S_i, state, data, priors)

_coupled(::BirthProposal) = nothing
_coupled(p::CoupledRJ) = p.coupled
_coupled(p::AuxiliaryMH) = p.coupled

_scale_vec(s::Real, names) = fill(float(s), length(names))
_scale_vec(s::NamedTuple, names) = [float(s[k]) for k in names]

# Centre of the contrast: T(fit_M) − T(fit_R), or zero when either fit is unavailable
function contrast_centre(model::LikelihoodModel, M::Vector{Int}, R::Vector{Int},
                         data::AbstractObservedData, priors::AbstractPriors)
    fM = moment_fit(model, M, data, priors)
    fR = moment_fit(model, R, data, priors)
    (isnothing(fM) || isnothing(fR)) && return zeros(length(param_scales(model)))
    return to_t(model, fM) .- to_t(model, fR)
end

function _contrast_dists(g::ContrastProposal, model, M, R, data, priors)
    names = keys(param_scales(model))
    s = _scale_vec(g.s, names)
    g isa ZeroStudentT && return [s[k] * TDist(g.ν) for k in eachindex(s)]
    μ = contrast_centre(model, M, R, data, priors)
    g isa FitNormal && return [Normal(μ[k], s[k]) for k in eachindex(s)]
    return [μ[k] + s[k] * TDist(g.ν) for k in eachindex(s)]
end

contrast_logpdf(g::ContrastProposal, model, M, R, u, data, priors) =
    sum(logpdf.(_contrast_dists(g, model, M, R, data, priors), u))

function contrast_draw(g::ContrastProposal, model, M, R, data, priors)
    d = _contrast_dists(g, model, M, R, data, priors)
    u = rand.(d)
    return u, sum(logpdf.(d, u))
end

"""
    coupled_birth_ratio(model, M, R, ρ_O, table_O, coupled, state, data, priors)

Draw a coupled birth splitting `table_O = M ∪ R` and return
`(log ratio without the link term, ρ_R, ρ_M)`. The state is not modified.
"""
function coupled_birth_ratio(model::LikelihoodModel, M::Vector{Int}, R::Vector{Int}, ρ_O::NamedTuple,
                             table_O::Vector{Int}, c::Coupled, state::AbstractMCMCState,
                             data::AbstractObservedData, priors::AbstractPriors)
    t_O = to_t(model, ρ_O)
    u, log_g = contrast_draw(c.contrast, model, M, R, data, priors)
    m, r = length(M), length(R)
    t_R = t_O .- (m / (m + r)) .* u
    t_M = t_O .+ (r / (m + r)) .* u
    ρ_R = from_t(model, t_R)
    ρ_M = from_t(model, t_M)
    lr = contribution_with(model, R, ρ_R, state, data, priors) + logjac(model, t_R) +
         contribution_with(model, M, ρ_M, state, data, priors) + logjac(model, t_M) -
         table_contribution(model, table_O, state, data, priors) - logjac(model, t_O) - log_g
    return lr, ρ_R, ρ_M
end

"""
    coupled_death_ratio(model, M, B, ρ_M, ρ_B, coupled, state, data, priors)

The coupled death merging `M` (its own cluster) into `B`; returns
`(log ratio without the link term, M ∪ B, ρ_O)`. The state is not modified.
"""
function coupled_death_ratio(model::LikelihoodModel, M::Vector{Int}, B::Vector{Int}, ρ_M::NamedTuple,
                             ρ_B::NamedTuple, c::Coupled, state::AbstractMCMCState,
                             data::AbstractObservedData, priors::AbstractPriors)
    t_M = to_t(model, ρ_M)
    t_B = to_t(model, ρ_B)
    m, r = length(M), length(B)
    t_O = (r .* t_B .+ m .* t_M) ./ (m + r)
    u = t_M .- t_B
    O = sorted_merge(M, B)
    ρ_O = from_t(model, t_O)
    lr = contribution_with(model, O, ρ_O, state, data, priors) + logjac(model, t_O) +
         contrast_logpdf(c.contrast, model, M, B, u, data, priors) -
         table_contribution(model, B, state, data, priors) - logjac(model, t_B) -
         table_contribution(model, M, state, data, priors) - logjac(model, t_M)
    return lr, O, ρ_O
end

# ============================================================================
# Posterior-informed link proposal
# ============================================================================

"""
    informed_link_proposal(model, i, M, R, table_O, ρ_O, state, data, priors, L, link)

Draw `c_i` from `InformedLink` and return `(j_star, corr)`, where `corr` is the
log prior ratio times the proposal ratio (the link weights cancel).
"""
function informed_link_proposal(model::LikelihoodModel, i::Int, M::Vector{Int}, R::Vector{Int},
                                table_O::Vector{Int}, ρ_O::NamedTuple, state::AbstractMCMCState,
                                data::AbstractObservedData, priors::AbstractPriors,
                                L::AbstractMatrix, link::InformedLink)
    L isa DDCRPMatrix || throw(ArgumentError("InformedLink needs a DDCRPMatrix"))
    dicts = cluster_param_dicts(state)
    w = view(L.wt, :, i)
    own = isempty(R)

    ρ̂ = moment_fit(model, M, data, priors)
    isnothing(ρ̂) && (ρ̂ = prior_mean_params(model, priors))
    blocks = [M]
    W = [sum(j -> w[j], M)]
    scores = [log(W[1]) + contribution_with(model, M, ρ̂, state, data, priors) -
              0.5 * length(param_scales(model)) * log(length(M))]
    if !own
        push!(blocks, R); push!(W, sum(j -> w[j], R))
        push!(scores, log(W[end]) + moving_set_loglik(model, M, ρ_O, R, state, data, priors))
    end
    for B in keys(first(dicts))
        B == table_O && continue
        push!(blocks, B); push!(W, sum(j -> w[j], B))
        push!(scores, log(W[end]) + moving_set_loglik(model, M, map(d -> d[B], dicts), B, state, data, priors))
    end
    replace!(s -> isnan(s) ? -Inf : s, scores)

    ls = link.τ .* scores
    mx = maximum(ls)
    sm = isfinite(mx) ? exp.(ls .- mx) : zeros(length(ls))
    tot = sum(sm)
    P = (1 - link.ε) .* (tot > 0 ? sm ./ tot : sm) .+ link.ε .* W ./ sum(W)
    tot > 0 || (P ./= sum(P))

    b = sample(1:length(P), Weights(P))
    j_star = _sample_within(w, blocks[b])
    b_old = own ? 1 : 2
    return j_star, log(P[b_old]) - log(W[b_old]) - log(P[b]) + log(W[b])
end

function _check_informed(link::LinkProposal, proposal::BirthProposal, fixed_dim::FixedDimensionProposal)
    link isa InformedLink || return nothing
    fixed_dim isa NoUpdate || proposal isa AuxiliaryMH ||
        throw(ArgumentError("InformedLink requires NoUpdate()"))
    isnothing(_coupled(proposal)) || throw(ArgumentError("InformedLink cannot be combined with a Coupled move"))
    return nothing
end
