# ============================================================================
# PSIS-LOO
# ============================================================================

@testset "PSIS-LOO" begin
    @testset "GPD fit recovers the shape" begin
        rng = MersenneTwister(1)
        for k_true in (0.0, 0.3, 0.8)
            x = sort(rand(rng, GeneralizedPareto(0.0, 1.0, k_true), 5000))
            k, σ = DistanceDependentCRP._gpd_fit(x)
            @test abs(k - k_true) < 0.08
            @test abs(σ - 1.0) < 0.15
        end
    end

    @testset "matches exact LOO in a conjugate normal model" begin
        # y_i ~ N(μ, 1), μ ~ N(0, 10²): posterior and leave-one-out predictive
        # densities are available in closed form.
        rng = MersenneTwister(2)
        n, S, τ² = 30, 4000, 100.0
        y = randn(rng, n) .+ 1.0
        post_var = 1 / (1 / τ² + n)
        μ = rand(rng, Normal(post_var * sum(y), sqrt(post_var)), S)
        ll = [logpdf(Normal(μ[s], 1.0), y[i]) for s in 1:S, i in 1:n]

        exact = map(1:n) do i
            v = 1 / (1 / τ² + n - 1)
            m = v * (sum(y) - y[i])
            logpdf(Normal(m, sqrt(1 + v)), y[i])
        end

        res = compute_psis_loo(ll)
        @test length(res.loo_i) == n
        @test all(res.k_hat .< 0.5)
        @test maximum(abs.(res.loo_i .- exact)) < 0.01
        @test res.elpd_loo ≈ sum(exact) atol = 0.05
    end

    @testset "flags heavy-tailed importance ratios" begin
        # Ratios drawn from a Pareto tail with shape 1.2 have infinite mean; the
        # previous moment estimator could never report k̂ above 0.5.
        rng = MersenneTwister(3)
        S = 4000
        ratios = rand(rng, GeneralizedPareto(1.0, 1.0, 1.2), S)
        ll = reshape(-log.(ratios), S, 1)
        res = compute_psis_loo(ll)
        @test res.k_hat[1] > 0.7
        @test isfinite(res.elpd_loo)
    end

    @testset "smoothed weights are truncated and normalised" begin
        rng = MersenneTwister(4)
        log_ratios = randn(rng, 2000) .* 3
        lw, k = DistanceDependentCRP._psis_smooth(log_ratios)
        @test DistanceDependentCRP._logsumexp(lw) ≈ 0.0 atol = 1e-10
        @test maximum(lw) <= 0.0
        # Smoothing replaces tail values by ordered quantiles, so it must not
        # reorder the weights.
        @test issorted(lw[sortperm(log_ratios)])
        @test isfinite(k)
    end

    @testset "too few draws to fit a tail" begin
        lw, k = DistanceDependentCRP._psis_smooth(randn(MersenneTwister(5), 20))
        @test isinf(k)
        @test DistanceDependentCRP._logsumexp(lw) ≈ 0.0 atol = 1e-10
    end
end
