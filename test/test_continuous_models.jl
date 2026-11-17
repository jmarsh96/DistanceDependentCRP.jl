# ============================================================================
# Tests for Continuous Models (Gamma)
# ============================================================================

@testset "Continuous Models" begin

    # ========================================================================
    # GammaClusterShapeMarg - Gamma Model with Marginalised Rate
    # ========================================================================

    @testset "GammaClusterShapeMarg - State Initialization" begin
        Random.seed!(123)
        n = 20
        # Simulate gamma data
        data_sim = simulate_gamma_data(n, [2.0, 4.0], [1.0, 2.0]; α=0.1, scale=5.0)
        data = ContinuousData(data_sim.y, data_sim.D)
        ddcrp_params = DDCRPParams(0.1, 5.0)
        priors = GammaClusterShapeMargPriors(2.0, 1.0, 2.0, 1.0)

        state = initialise_state(GammaClusterShapeMarg(), data, ddcrp_params, priors)

        @test state isa GammaClusterShapeMargState
        @test length(state.c) == n
        @test all(1 <= state.c[i] <= n for i in 1:n)
        @test !isempty(state.α_dict)

        # Check all observations are covered
        tables = table_vector(state.c)
        all_obs = sort(vcat(tables...))
        @test all_obs == 1:n

        # Check α parameters are positive
        for α_val in values(state.α_dict)
            @test α_val > 0
        end

        # Check dict keys are sorted
        for table in keys(state.α_dict)
            @test issorted(table)
        end

        # Test edge case: negative data should error
        y_negative = [-1.0, 2.0, 3.0]
        D_neg = zeros(3, 3)
        data_neg = ContinuousData(y_negative, D_neg)
        @test_throws AssertionError initialise_state(GammaClusterShapeMarg(), data_neg, ddcrp_params, priors)

        # Test edge case: zero in data should error
        y_zero = [0.0, 2.0, 3.0]
        data_zero = ContinuousData(y_zero, D_neg)
        @test_throws AssertionError initialise_state(GammaClusterShapeMarg(), data_zero, ddcrp_params, priors)
    end

    @testset "GammaClusterShapeMarg - Table Contribution" begin
        Random.seed!(123)
        n = 15
        data_sim = simulate_gamma_data(n, [3.0], [2.0]; α=0.05, scale=10.0)
        data = ContinuousData(data_sim.y, data_sim.D)
        ddcrp_params = DDCRPParams(0.05, 10.0)
        priors = GammaClusterShapeMargPriors(2.0, 1.0, 2.0, 1.0)

        state = initialise_state(GammaClusterShapeMarg(), data, ddcrp_params, priors)
        tables = table_vector(state.c)

        # Test table contribution for each table
        for table in tables
            contrib = table_contribution(GammaClusterShapeMarg(), table, state, data, priors)
            @test isfinite(contrib)
            @test contrib < 0  # Log-probability should be negative
        end

        # Test with single observation cluster
        c_single = collect(1:n)  # All self-loops
        α_dict_single = Dict{Vector{Int}, Float64}()
        for i in 1:n
            α_dict_single[[i]] = 2.0
        end
        state_single = GammaClusterShapeMargState(c_single, α_dict_single)

        for i in 1:n
            contrib = table_contribution(GammaClusterShapeMarg(), [i], state_single, data, priors)
            @test isfinite(contrib)
        end

        # Test with all in one cluster
        c_all = ones(Int, n)
        α_dict_all = Dict{Vector{Int}, Float64}([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15] => 3.0)
        state_all = GammaClusterShapeMargState(c_all, α_dict_all)
        contrib_all = table_contribution(GammaClusterShapeMarg(), collect(1:n), state_all, data, priors)
        @test isfinite(contrib_all)
    end

    @testset "GammaClusterShapeMarg - Marginal Likelihood" begin
        # Test that marginal likelihood integrates out β correctly
        Random.seed!(123)
        y = [1.5, 2.0, 2.5, 3.0]
        D = zeros(4, 4)
        data = ContinuousData(y, D)
        priors = GammaClusterShapeMargPriors(2.0, 1.0, 3.0, 2.0)

        table = [1, 2, 3, 4]
        α = 2.5
        α_dict = Dict{Vector{Int}, Float64}(table => α)
        c = ones(Int, 4)
        state = GammaClusterShapeMargState(c, α_dict)

        contrib = table_contribution(GammaClusterShapeMarg(), table, state, data, priors)

        # Marginal likelihood should be finite and include:
        # - Likelihood integrated over β
        # - Prior on α
        @test isfinite(contrib)

        # Test with different hyperpriors
        priors_weak = GammaClusterShapeMargPriors(0.1, 0.1, 0.1, 0.1)
        state_weak = GammaClusterShapeMargState(c, α_dict)
        contrib_weak = table_contribution(GammaClusterShapeMarg(), table, state_weak, data, priors_weak)
        @test isfinite(contrib_weak)

        # Test with strong priors
        priors_strong = GammaClusterShapeMargPriors(10.0, 5.0, 10.0, 5.0)
        state_strong = GammaClusterShapeMargState(c, α_dict)
        contrib_strong = table_contribution(GammaClusterShapeMarg(), table, state_strong, data, priors_strong)
        @test isfinite(contrib_strong)
    end

    @testset "GammaClusterShapeMarg - Posterior Calculation" begin
        Random.seed!(123)
        n = 12
        data_sim = simulate_gamma_data(n, [2.0, 4.0], [1.0, 1.0]; α=0.1, scale=5.0)
        data = ContinuousData(data_sim.y, data_sim.D)
        ddcrp_params = DDCRPParams(0.1, 5.0)
        priors = GammaClusterShapeMargPriors(2.0, 1.0, 2.0, 1.0)

        state = initialise_state(GammaClusterShapeMarg(), data, ddcrp_params, priors)
        log_DDCRP = precompute_log_ddcrp(decay, ddcrp_params.α, ddcrp_params.scale, data_sim.D)

        # Test posterior calculation
        post = posterior(GammaClusterShapeMarg(), data, state, priors, log_DDCRP)
        @test isfinite(post)

        # Use helper to check components
        check_posterior_components(GammaClusterShapeMarg(), state, data, priors, log_DDCRP)
    end

    @testset "GammaClusterShapeMarg - Shape Parameter Updates" begin
        Random.seed!(123)
        n = 10
        data_sim = simulate_gamma_data(n, [3.0], [1.0]; α=0.05, scale=10.0)
        data = ContinuousData(data_sim.y, data_sim.D)
        priors = GammaClusterShapeMargPriors(2.0, 1.0, 2.0, 1.0)

        # Initialize with known state
        c = ones(Int, n)
        α_dict = Dict{Vector{Int}, Float64}([1,2,3,4,5,6,7,8,9,10] => 3.0)
        state = GammaClusterShapeMargState(c, α_dict)

        α_before = copy(state.α_dict)

        # Update α parameters (MH on log-scale)
        update_α!(GammaClusterShapeMarg(), state, data, priors; prop_sd=0.5)

        # Check that update was attempted (may or may not accept)
        @test !isempty(state.α_dict)

        # Check positivity maintained
        for α_val in values(state.α_dict)
            @test α_val > 0
        end

        # Run multiple updates to check it doesn't crash
        for _ in 1:10
            update_α!(GammaClusterShapeMarg(), state, data, priors; prop_sd=0.3)
            @test all(α > 0 for α in values(state.α_dict))
        end
    end

    @testset "GammaClusterShapeMarg - Edge Cases" begin
        Random.seed!(123)
        priors = GammaClusterShapeMargPriors(2.0, 1.0, 2.0, 1.0)

        # Single observation cluster
        y_single = [2.5]
        D_single = zeros(1, 1)
        data_single = ContinuousData(y_single, D_single)
        c_single = [1]
        α_dict_single = Dict{Vector{Int}, Float64}([1] => 2.0)
        state_single = GammaClusterShapeMargState(c_single, α_dict_single)

        contrib_single = table_contribution(GammaClusterShapeMarg(), [1], state_single, data_single, priors)
        @test isfinite(contrib_single)

        # Very small shape (high variance)
        y_small = rand(Gamma(0.5, 2.0), 10)
        D_small = zeros(10, 10)
        data_small = ContinuousData(y_small, D_small)
        c_small = ones(Int, 10)
        α_dict_small = Dict{Vector{Int}, Float64}([1,2,3,4,5,6,7,8,9,10] => 0.2)
        state_small = GammaClusterShapeMargState(c_small, α_dict_small)

        contrib_small = table_contribution(GammaClusterShapeMarg(), collect(1:10), state_small, data_small, priors)
        @test isfinite(contrib_small)

        # Very large shape (low variance)
        y_large = rand(Gamma(50.0, 0.1), 10)
        D_large = zeros(10, 10)
        data_large = ContinuousData(y_large, D_large)
        α_dict_large = Dict{Vector{Int}, Float64}([1,2,3,4,5,6,7,8,9,10] => 50.0)
        state_large = GammaClusterShapeMargState(c_small, α_dict_large)

        contrib_large = table_contribution(GammaClusterShapeMarg(), collect(1:10), state_large, data_large, priors)
        @test isfinite(contrib_large)
    end

    @testset "GammaClusterShapeMarg - cluster statistics" begin
        const_DD = DistanceDependentCRP
        model = GammaClusterShapeMarg()
        Random.seed!(7)
        n = 18
        data_sim = simulate_gamma_data(n, [2.0, 5.0], [1.0, 2.0]; α=0.2, scale=5.0)
        data = ContinuousData(data_sim.y, data_sim.D)
        ddcrp = DDCRPParams(0.2, 5.0)
        priors = GammaClusterShapeMargPriors(2.0, 0.5, 2.0, 0.5)
        y = data_sim.y

        # Reference: the closed form written out from the raw observations
        function ref(table, α)
            m = length(table); sy = sum(y[table]); sl = sum(log.(y[table]))
            (α - 1) * sl - m * const_DD.loggamma(α) + const_DD.loggamma(m * α + priors.β_a) -
                const_DD.loggamma(priors.β_a) + priors.β_a * log(priors.β_b) -
                (m * α + priors.β_a) * log(sy + priors.β_b) + logpdf(Gamma(priors.α_a, 1 / priors.α_b), α)
        end

        state = initialise_state(model, data, ddcrp, priors)
        for (t, α) in state.α_dict
            @test table_contribution(model, t, state, data, priors) ≈ ref(t, α)
            @test haskey(state.stats, t)
            @test table_contribution(model, t, state, data, priors) ≈ ref(t, α)   # memo hit
        end
        @test length(state.logy) == n

        # Two-argument constructor: log y filled lazily
        t_all = collect(1:n)
        state2 = GammaClusterShapeMargState(ones(Int, n), Dict{Vector{Int},Float64}(t_all => 3.0))
        @test isempty(state2.logy)
        @test table_contribution(model, t_all, state2, data, priors) ≈ ref(t_all, 3.0)
        @test length(state2.logy) == n

        # Moving-set likelihood from statistics equals the generic difference and the reference
        M, B = [2, 5, 9], [1, 3, 4, 7]
        ρ = (α = 2.3,)
        keys_before = Set(keys(state.α_dict))
        fast = const_DD.moving_set_loglik(model, M, ρ, B, state, data, priors)
        slow = contribution_with(model, const_DD.sorted_merge(M, B), ρ, state, data, priors) -
               contribution_with(model, B, ρ, state, data, priors)
        @test fast ≈ slow
        @test fast ≈ ref(sort(vcat(M, B)), ρ.α) - ref(B, ρ.α)
        @test Set(keys(state.α_dict)) == keys_before
        sM = suff_stats(model, M, state, data)
        @test sM.n == 3 && sM.sy ≈ sum(y[M]) && sM.slogy ≈ sum(log.(y[M]))
        @test stats_contribution(model, sM, ρ, priors) ≈ ref(M, ρ.α)
        @test isnothing(suff_stats(PoissonClusterRates(), [1], state, data))

        # Pool log weights with sM equal contribution_with per auxiliary
        q = LogNormalMomentMatch(0.5)
        Random.seed!(3)
        aux, a, _ = const_DD._auxiliary_pool(model, q, 3, M, sM, ρ, true, state, data, priors)
        for t in 1:3
            lq = birth_params_logpdf(model, q, aux[t], M, state, data, priors)
            @test a[t] ≈ contribution_with(model, M, aux[t], state, data, priors) - lq
        end

        # Eviction drops tables that are not clusters
        contribution_with(model, [1, 2, 3], ρ, state, data, priors)
        @test haskey(state.stats, [1, 2, 3])
        update_params!(model, state, data, priors, table_vector(state.c), zeros(n, n),
                       MCMCOptions(infer_params = Dict(:α => false)))
        @test all(haskey(state.α_dict, k) for k in keys(state.stats))

        # Chain equality: flushing the memo before every update changes nothing
        L = precompute_log_ddcrp(ddcrp.decay_fn, ddcrp.α, ddcrp.scale, data_sim.D)
        opts = MCMCOptions(prop_sds = Dict(:α => 0.5))
        function run(p, flush)
            Random.seed!(99)
            st = initialise_state(model, data, ddcrp, priors)
            cs = Vector{Vector{Int}}()
            for _ in 1:30
                flush && empty!(st.stats)
                update_params!(model, st, data, priors, table_vector(st.c), L, opts)
                for i in 1:n
                    flush && empty!(st.stats)
                    const_DD.update_c_i!(model, i, st, data, priors, L, p, NoUpdate(), PriorLink(),
                                         const_DD.StandardUpdate())
                end
                push!(cs, copy(st.c))
            end
            return cs, st.α_dict
        end
        for p in (AuxiliaryGibbs(q, 2), AuxiliaryMH(q, 2), q)
            @test run(p, false) == run(p, true)
        end
    end

end
