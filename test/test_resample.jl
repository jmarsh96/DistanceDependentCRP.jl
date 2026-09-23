# ============================================================================
# Resample fixed-dimension proposal
# ============================================================================
# A fixed-dimension move takes the moving set S_i from `table_depl` to
# `table_aug`. The reverse move takes it back, so it scores the current
# parameters under the proposal built on the CURRENT memberships (`table_depl`
# with S_i, `table_aug` without it). Scoring them on the proposed memberships
# instead leaves the Hastings ratio wrong for any data-dependent inner proposal.
# That error biases the posterior over partitions while leaving P(K) and the
# rate posteriors almost untouched, so it is tested at the level of the ratio and
# against an exactly enumerated posterior.

using SpecialFunctions: loggamma

@testset "Resample fixed-dimension proposal" begin

    @testset "Hastings ratio scores the current values on the current memberships" begin
        Random.seed!(21)
        n = 10
        c = collect(1:n)
        c[2:5] .= 1
        c[6:10] .= 6
        tables = [sort(t) for t in table_vector(c)]
        table_depl = only(t for t in tables if 1 in t)
        table_aug  = only(t for t in tables if 6 in t)
        state = PoissonClusterRatesState(c, Dict(table_depl => 3.0, table_aug => 9.0))
        y = [1, 2, 3, 2, 1, 8, 12, 10, 9, 11]
        data = CountData(y, zeros(n, n))
        priors = PoissonClusterRatesPriors(1.0, 0.1)
        model = PoissonClusterRates()
        S_i = [4, 5]
        remaining = sort(setdiff(table_depl, S_i))
        augmented = sort(vcat(table_aug, S_i))

        # For PoissonClusterRates the PriorProposal draws from the conjugate
        # posterior given the members, so it is data-dependent too.
        for inner in (PriorProposal(), NormalMomentMatch(0.5), LogNormalMomentMatch(0.5))
            vd, va, lpr = DistanceDependentCRP.fixed_dim_param(
                model, Val(:λ), Resample(inner), S_i, table_depl, table_aug, state, data, priors)
            lq(v, members) = DistanceDependentCRP.birth_param_logpdf(
                model, Val(:λ), inner, v, members, state, data, priors)
            expected = (lq(3.0, table_depl) + lq(9.0, table_aug)) - (lq(vd, remaining) + lq(va, augmented))
            @test lpr ≈ expected
            # The pre-fix ratio scored the current values on the proposed memberships.
            wrong = (lq(3.0, remaining) + lq(9.0, augmented)) - (lq(vd, remaining) + lq(va, augmented))
            @test !(lpr ≈ wrong)
        end
    end

    @testset "Resample recovers the exactly enumerated partition posterior" begin
        # Four observations, so the posterior over partitions can be enumerated
        # over all 4^4 link vectors using the conjugate marginal likelihood. The
        # rates are well separated, so moment-matched centres differ markedly
        # between the current and proposed memberships. Before the fix the
        # moment-matched Resample chains sat about 0.011 from the exact posterior
        # in total variation, against 0.002 for NoUpdate at this chain length.
        n = 4
        y = [0, 2, 9, 14]
        x = [0.0, 0.5, 1.0, 1.5]
        D = [abs(x[i] - x[j]) for i in 1:n, j in 1:n]
        α, s, a, b = 1.0, 1.0, 1.0, 0.1
        ddcrp = DDCRPParams(α, s, DistanceDependentCRP.exp_decay, 1.0, 0.01, 1.0, 0.01)
        w(i, j) = i == j ? α : exp(-s * D[i, j])
        logml(m) = a * log(b) - loggamma(a) + loggamma(a + sum(y[m])) -
                   (a + sum(y[m])) * log(b + length(m)) - sum(loggamma.(y[m] .+ 1))

        exact = Dict{Vector{Int},Float64}()
        for cc in Iterators.product(ntuple(_ -> 1:n, n)...)
            cv = collect(cc)
            z = compute_table_assignments(cv)
            lp = sum(log(w(i, cv[i])) for i in 1:n) + sum(logml(findall(==(k), z)) for k in unique(z))
            exact[z] = get(exact, z, 0.0) + exp(lp)
        end
        total = sum(values(exact))
        foreach(k -> exact[k] /= total, keys(exact))

        N, burn = 400_000, 20_000
        opts = MCMCOptions(n_samples = N, verbose = false, track_diagnostics = false,
                           infer_params = Dict(:α_ddcrp => false, :s_ddcrp => false))
        function tv_to_exact(fixed_dim)
            Random.seed!(7)
            smp = mcmc(PoissonClusterRates(), CountData(y, D), ddcrp, PoissonClusterRatesPriors(a, b),
                       PriorProposal(); fixed_dim_proposal = fixed_dim, opts = opts)
            emp = Dict{Vector{Int},Float64}()
            for t in (burn + 1):N
                z = compute_table_assignments(smp.c[t, :])
                emp[z] = get(emp, z, 0.0) + 1
            end
            0.5 * sum(abs(get(emp, z, 0.0) / (N - burn) - p) for (z, p) in exact)
        end

        @test tv_to_exact(NoUpdate()) < 0.006
        @test tv_to_exact(Resample(NormalMomentMatch(0.25))) < 0.006
        @test tv_to_exact(Resample(LogNormalMomentMatch(0.25))) < 0.006
    end
end
