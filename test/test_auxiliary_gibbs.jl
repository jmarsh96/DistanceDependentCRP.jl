# ============================================================================
# Auxiliary-variable Gibbs update (ddCRP Algorithm 8)
# ============================================================================

const DD = DistanceDependentCRP

# Exact partition posterior by enumerating all n^n link vectors
function exact_partition_posterior(y, D, ddcrp, priors_marg)
    n = length(y)
    log_DDCRP = precompute_log_ddcrp(ddcrp.decay_fn, ddcrp.α, ddcrp.scale, D)
    data = CountData(y, D)
    state = PoissonClusterRatesMargState(ones(Int, n))
    post = Dict{Vector{Int}, Float64}()
    for idx in CartesianIndices(ntuple(_ -> n, n))
        c = collect(Tuple(idx))
        lp = ddcrp_contribution(c, log_DDCRP)
        lp += sum(table_contribution(PoissonClusterRatesMarg(), t, state, data, priors_marg)
                  for t in table_vector(c))
        z = compute_table_assignments(c)
        post[z] = get(post, z, 0.0) + exp(lp)
    end
    total = sum(values(post))
    return Dict(z => w / total for (z, w) in post)
end

function partition_frequencies(c_samples)
    freq = Dict{Vector{Int}, Float64}()
    for r in 1:size(c_samples, 1)
        z = compute_table_assignments(c_samples[r, :])
        freq[z] = get(freq, z, 0.0) + 1.0
    end
    return Dict(z => f / size(c_samples, 1) for (z, f) in freq)
end

tv_distance(p, q) = 0.5 * sum(abs(get(p, z, 0.0) - get(q, z, 0.0)) for z in union(keys(p), keys(q)))

@testset "Auxiliary Gibbs" begin
    y = [1, 2, 1, 8, 9]
    x = [0.0, 0.5, 1.0, 3.0, 3.5]
    D = [abs(a - b) for a in x, b in x]
    ddcrp = DDCRPParams(1.0, 1.0, DD.exp_decay, 1.0, 0.01, 1.0, 0.01)
    priors = PoissonClusterRatesPriors(2.0, 0.5)
    exact = exact_partition_posterior(y, D, ddcrp, PoissonClusterRatesMargPriors(2.0, 0.5))

    @testset "matches the exact partition posterior (q = $qname, m = $m)" for
            (qname, q) in [("prior", FixedDistributionProposal(Gamma(2.0, 2.0))),
                           ("LNMM", LogNormalMomentMatch(0.5))],
            m in (1, 3)
        Random.seed!(2024 + m)
        opts = MCMCOptions(n_samples = 60_000, verbose = false, track_diagnostics = false,
                           infer_params = Dict(:α_ddcrp => false, :s_ddcrp => false))
        s = mcmc(PoissonClusterRates(), CountData(y, D), ddcrp, priors, AuxiliaryGibbs(q, m);
                 opts = opts)
        emp = partition_frequencies(s.c[5_001:end, :])
        @test tv_distance(emp, exact) < 0.03
    end

    @testset "dicts stay in step with the links" begin
        Random.seed!(7)
        n = 30
        xs = sort(rand(n) .* 5)
        Dn = [abs(a - b) for a in xs, b in xs]
        yn = rand(Poisson(4), n)
        data = CountData(yn, Dn)
        state = initialise_state(PoissonClusterRates(), data, ddcrp, priors)
        log_DDCRP = precompute_log_ddcrp(ddcrp.decay_fn, ddcrp.α, ddcrp.scale, Dn)
        p = AuxiliaryGibbs(LogNormalMomentMatch(0.5), 3)
        ok = true
        for _ in 1:20, i in 1:n
            DD.update_c_i!(PoissonClusterRates(), i, state, data, priors, log_DDCRP, p,
                           NoUpdate(), UniformLink(), DD.StandardUpdate())
            ok &= Set(keys(state.λ_dict)) == Set(sort.(table_vector(state.c)))
        end
        @test ok
    end

    @testset "init_c sets the starting partition" begin
        Random.seed!(3)
        c0 = collect(1:length(y))
        opts = MCMCOptions(n_samples = 2, verbose = false, track_diagnostics = false,
                           infer_params = Dict(:c => false, :α_ddcrp => false))
        s = mcmc(PoissonClusterRates(), CountData(y, D), ddcrp, priors,
                 AuxiliaryGibbs(LogNormalMomentMatch(0.5), 2); opts = opts, init_c = c0)
        @test s.c[1, :] == c0
        @test s.c[2, :] == c0
    end

    @test_throws ArgumentError AuxiliaryGibbs(PriorProposal(), 0)

    @testset "moving-set likelihood: factorised and generic forms agree" begin
        Random.seed!(11)
        data = CountData(y, D)
        state = initialise_state(PoissonClusterRates(), data, ddcrp, priors)
        state.c .= [1, 1, 2, 4, 4]
        DD.set_partition!(PoissonClusterRates(), state, state.c, data, priors, PriorProposal())
        M, B = [3], [1, 2]
        ρ = (λ = 2.7,)
        fast = DD.moving_set_loglik(PoissonClusterRates(), M, ρ, B, state, data, priors)
        slow = contribution_with(PoissonClusterRates(), sort(vcat(M, B)), ρ, state, data, priors) -
               contribution_with(PoissonClusterRates(), B, ρ, state, data, priors)
        @test fast ≈ slow
    end
end

@testset "DDCRPMatrix" begin
    Random.seed!(5)
    xs = rand(12) .* 4
    D = [abs(a - b) for a in xs, b in xs]
    L = DDCRPMatrix(DD.exp_decay, 0.7, 1.3, D)
    @test Matrix(L) == precompute_log_ddcrp(DD.exp_decay, 0.7, 1.3, D)
    set_alpha!(L, 2.5)
    @test Matrix(L) ≈ precompute_log_ddcrp(DD.exp_decay, 2.5, 1.3, D)
    @test L.wt ≈ permutedims(exp.(Matrix(L)))
    @test L.offsum ≈ [sum(exp(L[i, j]) for j in 1:12 if j != i) for i in 1:12]
end

@testset "Metropolised Algorithm 8" begin
    y = [1, 2, 1, 8, 9]
    x = [0.0, 0.5, 1.0, 3.0, 3.5]
    D = [abs(a - b) for a in x, b in x]
    ddcrp = DDCRPParams(1.0, 1.0, DD.exp_decay, 1.0, 0.01, 1.0, 0.01)
    priors = PoissonClusterRatesPriors(2.0, 0.5)
    exact = exact_partition_posterior(y, D, ddcrp, PoissonClusterRatesMargPriors(2.0, 0.5))

    @testset "matches the exact partition posterior ($(nameof(typeof(link))), q = $qname, m = $m)" for
            link in (UniformLink(), PriorLink(), MetropolisedPriorLink()),
            (qname, q) in [("prior", FixedDistributionProposal(Gamma(2.0, 2.0))),
                           ("LNMM", LogNormalMomentMatch(0.5))],
            m in (1, 3)
        Random.seed!(4048 + m)
        opts = MCMCOptions(n_samples = 60_000, verbose = false, track_diagnostics = false,
                           infer_params = Dict(:α_ddcrp => false, :s_ddcrp => false),
                           link_proposal = link)
        s = mcmc(PoissonClusterRates(), CountData(y, D), ddcrp, priors, AuxiliaryMH(q, m);
                 opts = opts)
        emp = partition_frequencies(s.c[5_001:end, :])
        @test tv_distance(emp, exact) < 0.03
    end

    @testset "dicts stay in step with the links" begin
        Random.seed!(8)
        n = 30
        xs = sort(rand(n) .* 5)
        Dn = [abs(a - b) for a in xs, b in xs]
        data = CountData(rand(Poisson(4), n), Dn)
        state = initialise_state(PoissonClusterRates(), data, ddcrp, priors)
        L = DDCRPMatrix(ddcrp.decay_fn, ddcrp.α, ddcrp.scale, Dn)
        ok = true
        for link in (UniformLink(), PriorLink(), MetropolisedPriorLink()), _ in 1:10, i in 1:n
            DD.update_c_i!(PoissonClusterRates(), i, state, data, priors, L,
                           AuxiliaryMH(LogNormalMomentMatch(0.5), 3), NoUpdate(), link, DD.StandardUpdate())
            ok &= Set(keys(state.λ_dict)) == Set(sort.(table_vector(state.c)))
        end
        @test ok
    end

    @test_throws ArgumentError AuxiliaryMH(PriorProposal(), 0)
    @test_throws ArgumentError AuxiliaryMH(AuxiliaryGibbs(PriorProposal(), 1), 1)
end
