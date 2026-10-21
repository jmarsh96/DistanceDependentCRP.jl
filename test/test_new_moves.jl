# ============================================================================
# Coupled birth/death, informed link proposal and two-anchor split/merge
# (uses exact_partition_posterior, partition_frequencies, tv_distance from
# test_auxiliary_gibbs.jl)
# ============================================================================

const DDm = DistanceDependentCRP

const NM_Y = [1, 2, 1, 8, 9]
const NM_X = [0.0, 0.5, 1.0, 3.0, 3.5]
const NM_D = [abs(a - b) for a in NM_X, b in NM_X]
const NM_DDCRP = DDCRPParams(1.0, 1.0, DDm.exp_decay, 1.0, 0.01, 1.0, 0.01)
const NM_PRIORS = PoissonClusterRatesPriors(2.0, 0.5)

function nm_tv(birth, link; fdim = NoUpdate(), two_anchor = nothing, seed = 1)
    exact = exact_partition_posterior(NM_Y, NM_D, NM_DDCRP, PoissonClusterRatesMargPriors(2.0, 0.5))
    Random.seed!(seed)
    opts = MCMCOptions(n_samples = 60_000, verbose = false, track_diagnostics = true,
                       infer_params = Dict(:α_ddcrp => false, :s_ddcrp => false),
                       link_proposal = link, two_anchor = two_anchor)
    s, d = mcmc(PoissonClusterRates(), CountData(NM_Y, NM_D), NM_DDCRP, NM_PRIORS, birth;
                fixed_dim_proposal = fdim, opts = opts)
    return tv_distance(partition_frequencies(s.c[5_001:end, :]), exact), d
end

const NM_Q = LogNormalMomentMatch(0.5)
const NM_CONTRASTS = [FitStudentT(4.0, 1.0), ZeroStudentT(4.0, 1.0), FitNormal(1.0)]

@testset "Coupled birth/death" begin
    @testset "exact posterior, published RJ ($(nameof(typeof(g))))" for g in NM_CONTRASTS
        tv, d = nm_tv(CoupledRJ(NM_Q, Coupled(g, 0.5)), PriorLink())
        @test tv < 0.03
        @test d.coupled_birth_proposes > 0
    end
    @testset "exact posterior, RJ pool ($(nameof(typeof(g))), $(nameof(typeof(l))))" for
            g in NM_CONTRASTS, l in (PriorLink(), MetropolisedPriorLink())
        tv, _ = nm_tv(AuxiliaryMH(NM_Q, 1; coupled = Coupled(g, 0.5)), l)
        @test tv < 0.03
    end

    @testset "birth and death ratios are reciprocal" begin
        data = CountData(NM_Y, NM_D)
        state = initialise_state(PoissonClusterRates(), data, NM_DDCRP, NM_PRIORS)
        c = [1, 1, 2, 3, 5]                       # O = {1,2,3,4}, M = {3,4} once c_3 is cut
        set_partition!(PoissonClusterRates(), state, c, data, NM_PRIORS, PriorProposal())
        O = [1, 2, 3, 4]; M = [3, 4]; R = [1, 2]
        cp = Coupled(FitStudentT(4.0, 0.7), 1.0)
        Random.seed!(9)
        lr_b, ρ_R, ρ_M = DDm.coupled_birth_ratio(PoissonClusterRates(), M, R, (λ = state.λ_dict[O],),
                                                 O, cp, state, data, NM_PRIORS)
        # the state after the birth
        delete!(state.λ_dict, O); state.λ_dict[R] = ρ_R.λ; state.λ_dict[M] = ρ_M.λ
        lr_d, merged, ρ_O = DDm.coupled_death_ratio(PoissonClusterRates(), M, R, ρ_M, ρ_R, cp, state, data, NM_PRIORS)
        @test merged == O
        @test lr_b ≈ -lr_d
    end
end

@testset "Informed link proposal" begin
    @testset "exact posterior ($name)" for (name, birth) in
            [("published RJ", NM_Q), ("RJ pool m=1", AuxiliaryMH(NM_Q, 1)), ("RJ pool m=3", AuxiliaryMH(NM_Q, 3))]
        tv, _ = nm_tv(birth, InformedLink(0.1, 1.0))
        @test tv < 0.03
    end
    @test_throws ArgumentError nm_tv(CoupledRJ(NM_Q, Coupled(FitNormal(1.0))), InformedLink())
end

@testset "Two-anchor split/merge" begin
    @testset "exact posterior ($name, pilot = $pilot)" for (name, birth, link) in
            [("published RJ", NM_Q, PriorLink()), ("RJ pool", AuxiliaryMH(NM_Q, 1), PriorLink()),
             ("Gibbs", AuxiliaryGibbs(NM_Q, 2), UniformLink())], pilot in (false, true)
        ta = TwoAnchor(k = 2.0, attempts = 5, pilot = pilot, tilt = pilot ? 5.0 : 1.0, p_cross = 0.8)
        tv, d = nm_tv(birth, link; two_anchor = ta)
        @test tv < 0.03
        @test d.two_anchor.split_accepts > 0 && d.two_anchor.merge_accepts > 0
    end

    @testset "pilot groups" begin
        data = CountData(NM_Y, NM_D)
        state = initialise_state(PoissonClusterRates(), data, NM_DDCRP, NM_PRIORS)
        set_partition!(PoissonClusterRates(), state, [2, 3, 4, 5, 1], data, NM_PRIORS, PriorProposal())
        L = DDCRPMatrix(NM_DDCRP.decay_fn, NM_DDCRP.α, NM_DDCRP.scale, NM_D)
        ta = TwoAnchor(pilot = true)
        P = two_anchor_pilot(PoissonClusterRates(), state, data, NM_PRIORS, L, ta, 1, 5)
        @test [P.group[i] for i in 1:5] == [true, true, true, false, false]
        @test P.share[3] > P.share[1] && P.share[4] > P.share[5]     # the boundary is between 3 and 4
        @test isnothing(two_anchor_pilot(PoissonClusterRates(), state, data, NM_PRIORS, L, TwoAnchor(), 1, 5))
        # the fast pointwise likelihood agrees with a singleton contribution
        ρ = (λ = 2.5,)
        @test obs_loglik(PoissonClusterRates(), 4, ρ, data) ≈
              contribution_with(PoissonClusterRates(), [4], ρ, state, data, NM_PRIORS) -
              cluster_logprior(PoissonClusterRates(), ρ, NM_PRIORS)
        # the basic move is the pilot form with P = nothing
        U = collect(1:5); c2 = [1, 1, 2, 5, 4]
        @test two_anchor_logQ(state.c, c2, U, TwoAnchor(k = 2.0), nothing, true, L) ≈
              two_anchor_logQ(state.c, c2, U, 0.4, L)
        # with tilt = 1 the link prior and proposal terms cancel
        lw(c) = sum(L[i, c[i]] for i in U)
        @test lw(c2) - lw(state.c) + two_anchor_logQ(c2, state.c, U, ta, P, false, L) -
              two_anchor_logQ(state.c, c2, U, ta, P, true, L) ≈ 0 atol = 1e-10
    end

    @testset "split and merge ratios are reciprocal" begin
        y = [1, 2, 1, 7, 8, 9]
        x = [0.0, 0.4, 0.9, 3.0, 3.3, 3.9]
        D = [abs(a - b) for a in x, b in x]
        data = CountData(y, D)
        L = DDCRPMatrix(NM_DDCRP.decay_fn, NM_DDCRP.α, NM_DDCRP.scale, D)
        mk(c, λs) = (st = initialise_state(PoissonClusterRates(), data, NM_DDCRP, NM_PRIORS); st.c .= c;
                     empty!(st.λ_dict); for (t, λ) in zip(sort.(table_vector(c)), λs); st.λ_dict[t] = λ; end; st)
        # two cycles {1,2,3} and {4,5,6}; anchors a = 2, b = 5
        c_split = [2, 3, 1, 5, 6, 4]
        c_merged = [2, 3, 1, 5, 6, 2]              # 6 → 2 joins them; other links unchanged
        λA, λB = 1.6, 7.5
        x_split = mk(c_split, [λA, λB])
        x_merged = mk(c_merged, [λA])
        for ta in (TwoAnchor(k = 2.0), TwoAnchor(k = 2.0, pilot = true, tilt = 5.0, p_cross = 0.7))
            lr_merge, = two_anchor_logratio(PoissonClusterRates(), x_split, data, NM_PRIORS, L, NM_Q, ta, 2, 5, c_merged, nothing)
            lr_split, = two_anchor_logratio(PoissonClusterRates(), x_merged, data, NM_PRIORS, L, NM_Q, ta, 2, 5, c_split, (λ = λB,))
            @test lr_merge ≈ -lr_split
        end
        @test two_anchor_components(c_merged, collect(1:6))[1] == 1
        @test two_anchor_components(c_split, collect(1:6))[1] == 2
    end
end

# ----------------------------------------------------------------------------
# Finite-state detailed balance for the two-anchor kernel (discrete parameters)
# ----------------------------------------------------------------------------

struct ToyModel <: LikelihoodModel end
struct ToyPriors <: AbstractPriors end
struct ToyQ <: BirthProposal end
mutable struct ToyState <: DDm.AbstractMCMCState{Float64}
    c::Vector{Int}
    λ_dict::Dict{Vector{Int}, Float64}
end
const TOY_GRID = [1.0, 3.0, 8.0]
const TOY_PRIOR = [0.5, 0.3, 0.2]
DDm.cluster_param_dicts(s::ToyState) = (λ = s.λ_dict,)
DDm.table_contribution(::ToyModel, t::AbstractVector{Int}, s::ToyState, data, ::ToyPriors) =
    sum(logpdf(Poisson(s.λ_dict[t]), observations(data)[i]) for i in t) +
    log(TOY_PRIOR[findfirst(==(s.λ_dict[t]), TOY_GRID)])
# discrete q favouring the grid point nearest the set mean
toy_q(S, data) = (w = [exp(-abs(log(g) - log(mean(observations(data)[S]) + 0.5))) for g in TOY_GRID]; w ./ sum(w))
function DDm.sample_birth_params(::ToyModel, ::ToyQ, S::Vector{Int}, s::ToyState, data, ::ToyPriors)
    p = toy_q(S, data); k = rand(Categorical(p)); return (λ = TOY_GRID[k],), log(p[k])
end
DDm.birth_params_logpdf(::ToyModel, ::ToyQ, ρ::NamedTuple, S::Vector{Int}, s::ToyState, data, ::ToyPriors) =
    log(toy_q(S, data)[findfirst(==(ρ.λ), TOY_GRID)])
DDm.moment_fit(::ToyModel, S::Vector{Int}, data, ::ToyPriors) =
    length(S) < 2 ? nothing : (λ = TOY_GRID[argmin(abs.(TOY_GRID .- mean(observations(data)[S])))],)
DDm.prior_mean_params(::ToyModel, ::ToyPriors) = (λ = 3.0,)

@testset "Two-anchor detailed balance on a finite state space (pilot = $pilot, tilt = $tilt)" for
        (pilot, tilt) in ((false, 1.0), (true, 1.0), (true, 4.0))
    y = [0, 2, 7, 9]
    xs = [0.0, 1.0, 2.0, 3.0]
    D = [abs(a - b) for a in xs, b in xs]
    data = CountData(y, D)
    L = DDCRPMatrix(DDm.exp_decay, 0.8, 0.7, D)
    ta = TwoAnchor(k = 1.5, pilot = pilot, tilt = tilt, p_cross = 0.8)
    a, b = 1, 3
    n = 4

    toystate(c, λs) = ToyState(copy(c), Dict(t => λ for (t, λ) in zip(sort.(table_vector(c)), λs)))
    logπ(x) = sum(L[i, x.c[i]] for i in 1:n) +
              sum(DDm.table_contribution(ToyModel(), t, x, data, ToyPriors()) for t in keys(x.λ_dict))
    key(x) = (copy(x.c), sort([(t, λ) for (t, λ) in x.λ_dict]))

    # all states reachable from x under the pair-(a, b) kernel, with their probabilities
    function transitions(x)
        dicts = DDm.cluster_param_dicts(x)
        TA = DDm.find_table_for_customer(a, x.λ_dict)
        TB = DDm.find_table_for_customer(b, x.λ_dict)
        split = TA == TB
        U = split ? TA : DDm.sorted_merge(TA, TB)
        G = two_anchor_pilot(ToyModel(), x, data, ToyPriors(), L, ta, a, b)
        out = Tuple{ToyState, Float64}[]
        for idx in CartesianIndices(ntuple(_ -> length(U), length(U)))
            c_new = copy(x.c)
            for (k, i) in enumerate(U)
                c_new[i] = U[idx[k]]
            end
            ncomp, lab = two_anchor_components(c_new, U)
            (split ? (ncomp == 2 && lab[a] != lab[b]) : ncomp == 1) || continue
            lQ = two_anchor_logQ(x.c, c_new, U, ta, G, split, L)
            if split
                Bset = [i for i in U if lab[i] != lab[a]]
                for λB in TOY_GRID
                    lr, _, A, B, _ = two_anchor_logratio(ToyModel(), x, data, ToyPriors(), L, ToyQ(), ta, a, b, c_new, (λ = λB,))
                    xn = ToyState(c_new, copy(x.λ_dict)); delete!(xn.λ_dict, U)
                    xn.λ_dict[A] = x.λ_dict[U]; xn.λ_dict[B] = λB
                    push!(out, (xn, exp(lQ + DDm.birth_params_logpdf(ToyModel(), ToyQ(), (λ = λB,), Bset, x, data, ToyPriors()) + min(0.0, lr))))
                end
            else
                lr, = two_anchor_logratio(ToyModel(), x, data, ToyPriors(), L, ToyQ(), ta, a, b, c_new, nothing)
                xn = ToyState(c_new, copy(x.λ_dict)); delete!(xn.λ_dict, TA); delete!(xn.λ_dict, TB)
                xn.λ_dict[U] = x.λ_dict[TA]
                push!(out, (xn, exp(lQ + min(0.0, lr))))
            end
        end
        return out
    end
    prob(x, x2) = sum((pr for (xn, pr) in transitions(x) if key(xn) == key(x2)); init = 0.0)

    Random.seed!(17)
    worst = 0.0; checked = 0
    for _ in 1:60
        c = rand(1:n, n)
        K = length(table_vector(c))
        x = toystate(c, rand(TOY_GRID, K))
        for (xn, pr) in transitions(x)
            key(xn) == key(x) && continue
            back = prob(xn, x)
            lhs = logπ(x) + log(pr); rhs = logπ(xn) + log(back)
            worst = max(worst, abs(lhs - rhs)); checked += 1
        end
    end
    @test checked > 100
    @test worst < 1e-8

    # the sampler draws its proposals with the probabilities above
    x = toystate([2, 1, 4, 3], [1.0, 8.0])              # {1,2} and {3,4}: a merge attempt
    Random.seed!(3)
    counts = Dict{Any, Int}(); N = 40_000
    for _ in 1:N
        xx = toystate(x.c, [1.0, 8.0])
        TAx = DDm.find_table_for_customer(a, xx.λ_dict); TBx = DDm.find_table_for_customer(b, xx.λ_dict)
        U = DDm.sorted_merge(TAx, TBx)
        G = two_anchor_pilot(ToyModel(), xx, data, ToyPriors(), L, ta, a, b)
        c_new = two_anchor_draw(xx.c, U, ta, G, false, L)
        two_anchor_components(c_new, U)[1] == 1 || continue
        k = key(ToyState(c_new, Dict(U => 1.0)))
        counts[k] = get(counts, k, 0) + 1
    end
    prop_expect = Dict{Any, Float64}()
    U = [1, 2, 3, 4]
    G = two_anchor_pilot(ToyModel(), x, data, ToyPriors(), L, ta, a, b)
    for (xn, _) in transitions(x)
        prop_expect[key(ToyState(xn.c, Dict(U => 1.0)))] = exp(two_anchor_logQ(x.c, xn.c, U, ta, G, false, L))
    end
    @test maximum(abs(get(counts, k, 0) / N - v) for (k, v) in prop_expect) < 0.005
end

@testset "In-loop thinning" begin
    run(thin) = (Random.seed!(21); mcmc(PoissonClusterRates(), CountData(NM_Y, NM_D), NM_DDCRP, NM_PRIORS, NM_Q;
        opts = MCMCOptions(n_samples = 101, verbose = false, track_diagnostics = false, thin = thin,
                           link_proposal = PriorLink())))
    full = run(1); thinned = run(4)
    @test size(thinned.c, 1) == cld(101, 4)
    @test thinned.c == full.c[1:4:end, :]
    @test thinned.α_ddcrp == full.α_ddcrp[1:4:end]
end
