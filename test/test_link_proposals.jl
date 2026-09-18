# ============================================================================
# Link proposals
# ============================================================================

@testset "Link proposals" begin
    @testset "PriorLink samples proportional to the ddCRP weights" begin
        Random.seed!(11)
        n = 6
        # Arbitrary row of log weights, including one negligible entry of the
        # kind a windowed decay function produces.
        logw = [0.0 -1.0 -2.0 -30.0 0.5 -0.25]
        log_DDCRP = repeat(logw, n, 1)
        draws = [DistanceDependentCRP.propose_link(1, 2, log_DDCRP, PriorLink())[1] for _ in 1:200_000]
        emp = [count(==(j), draws) / length(draws) for j in 1:n]
        w = exp.(vec(logw)); w ./= sum(w)
        @test maximum(abs.(emp .- w)) < 0.005
        # The prior ratio is cancelled by the proposal ratio.
        @test all(DistanceDependentCRP.propose_link(1, 2, log_DDCRP, PriorLink())[2] == 0.0 for _ in 1:10)
    end

    @testset "UniformLink is uniform and returns the prior ratio" begin
        Random.seed!(12)
        n = 5
        log_DDCRP = [log(abs(i - j) + 1.0) for i in 1:n, j in 1:n]
        draws = [DistanceDependentCRP.propose_link(2, 3, log_DDCRP, UniformLink()) for _ in 1:100_000]
        emp = [count(d -> d[1] == j, draws) / length(draws) for j in 1:n]
        @test maximum(abs.(emp .- 1 / n)) < 0.01
        @test all(d -> d[2] ≈ log_DDCRP[2, d[1]] - log_DDCRP[2, 3], draws)
    end

    @testset "both link proposals target the same posterior" begin
        # Windowed decay: only near neighbours carry weight, which is the case
        # where a uniform draw wastes most proposals. Both samplers must still
        # give the same posterior over the number of clusters.
        Random.seed!(13)
        n = 30
        x = collect(range(0, 3; length = n))
        D = [abs(x[i] - x[j]) for i in 1:n, j in 1:n]
        D = [D[i, j] <= 0.4 ? 0.0 : 30.0 for i in 1:n, j in 1:n]   # hop-window style
        y = vcat(rand(Poisson(2), n ÷ 2), rand(Poisson(12), n - n ÷ 2))
        ddcrp = DDCRPParams(1.0, 1.0, DistanceDependentCRP.exp_decay, 1.0, 0.01, 1.0, 0.01)
        priors = PoissonClusterRatesPriors(1.0, 0.1)

        function run(link)
            Random.seed!(99)
            opts = MCMCOptions(n_samples = 30_000, verbose = false, track_diagnostics = true,
                               infer_params = Dict(:α_ddcrp => true, :s_ddcrp => false),
                               link_proposal = link)
            s, d = mcmc(PoissonClusterRates(), CountData(y, D), ddcrp, priors,
                        LogNormalMomentMatch(0.5); fixed_dim_proposal = NoUpdate(), opts = opts)
            keep = 10_001:5:30_000
            (K = calculate_n_clusters(s.c[keep, :]), acc = acceptance_rates(d))
        end

        u = run(UniformLink()); p = run(PriorLink())
        # Same posterior mean K, within Monte Carlo error of these chain lengths.
        @test abs(mean(u.K) - mean(p.K)) < 1.0
        # Same posterior distribution: total variation distance between the two
        # P(K) histograms stays small.
        ks = union(unique(u.K), unique(p.K))
        tv = 0.5 * sum(abs(count(==(k), u.K) / length(u.K) - count(==(k), p.K) / length(p.K)) for k in ks)
        @test tv < 0.15
        # And the point of the change: far more proposals are accepted.
        @test p.acc.overall > 3 * u.acc.overall
    end
end
