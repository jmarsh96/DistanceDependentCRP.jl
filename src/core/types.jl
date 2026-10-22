# ============================================================================
# Abstract Type Hierarchy for DDCRP Models
# ============================================================================

"""
    LikelihoodModel

Abstract supertype for all likelihood models in the DDCRP framework.
Each model variant defines how cluster data contributes to the likelihood.
"""
abstract type LikelihoodModel end

"""
    PoissonModel <: LikelihoodModel

Abstract type for Poisson likelihood models.
Concrete subtypes: `PoissonClusterRates`, `PoissonClusterRatesMarg`, `PoissonPopulationRates`
"""
abstract type PoissonModel <: LikelihoodModel end

"""
    BinomialModel <: LikelihoodModel

Abstract type for Binomial likelihood models.
Concrete subtypes: `BinomialClusterProb`, `BinomialClusterProbMarg`
"""
abstract type BinomialModel <: LikelihoodModel end

"""
    GammaModel <: LikelihoodModel

Abstract type for Gamma likelihood models.
Concrete subtypes: `GammaClusterShapeMarg`
"""
abstract type GammaModel <: LikelihoodModel end

# ============================================================================
# Abstract Types for State, Priors, and Samples
# ============================================================================

"""
    AbstractMCMCState{T<:Real}

Abstract supertype for MCMC state containers.
Each model variant has its own state type.
"""
abstract type AbstractMCMCState{T<:Real} end

"""
    AbstractPriors

Abstract supertype for prior specifications.
Allows type-dispatched prior handling and validation.
"""
abstract type AbstractPriors end

"""
    AbstractMCMCSamples

Abstract supertype for MCMC output containers.
"""
abstract type AbstractMCMCSamples end

# ============================================================================
# Birth Proposals for RJMCMC
# ============================================================================

"""
    BirthProposal

Abstract supertype for RJMCMC birth proposal distributions.
Controls how new cluster parameters are proposed when clusters split.
Proposal objects are passed directly to `mcmc` and carry their own configuration.
"""
abstract type BirthProposal end

"""
    PriorProposal <: BirthProposal

Sample new cluster parameters from the prior distribution.
"""
struct PriorProposal <: BirthProposal end

"""
    ConjugateProposal <: BirthProposal

Marker type indicating the model has conjugate cluster parameters.
When used, `update_c!` dispatches to Gibbs sampling for assignments
instead of RJMCMC, and cluster parameters are resampled from their
conjugate posteriors after assignment updates.
"""
struct ConjugateProposal <: BirthProposal end

"""
    MomentMatchedProposal <: BirthProposal

Abstract supertype for data-informed birth proposals that use
empirical moments of the moving set to construct the proposal distribution.
"""
abstract type MomentMatchedProposal <: BirthProposal end

"""
    NormalMomentMatch <: MomentMatchedProposal

Sample new cluster parameters from truncated Normal centered at empirical mean.

# Fields
- `σ::Vector{Float64}`: One proposal std per cluster parameter
"""
struct NormalMomentMatch <: MomentMatchedProposal
    σ::Vector{Float64}
end
NormalMomentMatch(σ::Float64) = NormalMomentMatch([σ])
NormalMomentMatch(σs::Float64...) = NormalMomentMatch(collect(σs))

"""
    InverseGammaMomentMatch <: MomentMatchedProposal

Fit InverseGamma to data in moving set via method of moments.
Falls back to prior if moment matching fails.

# Fields
- `min_size::Int`: Minimum cluster size to attempt moment matching
"""
struct InverseGammaMomentMatch <: MomentMatchedProposal
    min_size::Int
end
InverseGammaMomentMatch() = InverseGammaMomentMatch(3)

"""
    LogNormalMomentMatch <: MomentMatchedProposal

Sample on log-scale using moment-matched LogNormal proposal.
For each parameter, proposes log(θ) ~ Normal(log(θ_est), σ) where
θ_est is a moment-based estimate.

# Fields
- `σ::Vector{Float64}`: One proposal std per cluster parameter (on log-scale)
- `min_size::Int`: Minimum cluster size for moment estimation
"""
struct LogNormalMomentMatch <: MomentMatchedProposal
    σ::Vector{Float64}
    min_size::Int
end
LogNormalMomentMatch(σ::Float64; min_size::Int=2) = LogNormalMomentMatch([σ], min_size)
LogNormalMomentMatch(σs::Vector{Float64}; min_size::Int=2) = LogNormalMomentMatch(σs, min_size)

"""
    FixedDistributionProposal <: BirthProposal

Sample new cluster parameters from user-specified fixed distributions.

# Fields
- `dists::Vector{UnivariateDistribution}`: One distribution per cluster parameter
"""
struct FixedDistributionProposal <: BirthProposal
    dists::Vector{UnivariateDistribution}
end
FixedDistributionProposal(d::UnivariateDistribution) = FixedDistributionProposal([d])

"""
    MixedProposal{T<:NamedTuple} <: BirthProposal

Compose per-parameter birth proposals. Each cluster parameter can use a different
proposal strategy. The `proposals` field is a NamedTuple mapping parameter names
(e.g. `:λ`, `:α`) to individual `BirthProposal` instances.

Dispatches to `sample_birth_param` and `birth_param_logpdf` for each parameter,
which are implemented per (model, parameter, proposal) combination in each model file.

# Example
```julia
MixedProposal(
    λ = LogNormalMomentMatch(0.5),
    α = NormalMomentMatch(0.5)
)
```
"""
struct MixedProposal{T<:NamedTuple} <: BirthProposal
    proposals::T
end
MixedProposal(; kwargs...) = MixedProposal(NamedTuple(kwargs))

"""
    AuxiliaryGibbs{P<:BirthProposal} <: BirthProposal

Select the auxiliary-variable Gibbs update for the links: the ddCRP form of
Algorithm 8 of Neal (2000). Each `c_i` is drawn from its full conditional over
all `n` values, with `m` auxiliary parameters for the moving set drawn from
`proposal` and weighted by prior/proposal. With `proposal = PriorProposal()`
this is Neal's original algorithm. The fixed-dimension and link proposals
passed to `mcmc` are ignored.

With a finite `skip`, the existing blocks of `z(c_{-i})` whose total link weight
from `i` is below `exp(-skip)` times the largest block weight (`M`'s included)
are left out of the Gibbs draw, so the moving set is scored only under the near
blocks. The far blocks are reached by a Metropolis-Hastings move between the
near and far blocks, proposed from the link weights, so the update stays exact.
This pays when the prior confines links to a neighbourhood (a windowed decay).

# Fields
- `proposal::P`: Density `q(· | y_M)` for the auxiliary parameters
- `m::Int`: Number of auxiliary parameters
- `skip::Float64`: Log weight gap beyond which a block is far (`Inf`: none; keyword)

# Example
```julia
AuxiliaryGibbs(MixedProposal(ξ = NormalMomentMatch(2.0), ω = LogNormalMomentMatch(1.0)), 3)
```
"""
struct AuxiliaryGibbs{P<:BirthProposal} <: BirthProposal
    proposal::P
    m::Int
    skip::Float64
    function AuxiliaryGibbs(proposal::P, m::Int; skip::Real = Inf) where {P<:BirthProposal}
        m >= 1 || throw(ArgumentError("m must be at least 1"))
        skip > 0 || throw(ArgumentError("skip must be positive"))
        proposal isa AuxiliaryGibbs && throw(ArgumentError("nested AuxiliaryGibbs"))
        new{P}(proposal, m, Float64(skip))
    end
end

# ============================================================================
# Coupled-parameter birth/death
# ============================================================================

"""
    ContrastProposal

Density `g(u)` of the contrast `u = t_M − t_R` between the two daughter
parameters of a coupled birth, on the unconstrained scale `t = T(ρ)` of
`param_scales`. Scales `s` are a number (every coordinate) or a NamedTuple
keyed by parameter name.
"""
abstract type ContrastProposal end

"""
    FitStudentT(ν, s)

Per-coordinate Student-t with `ν` degrees of freedom and scale `s`, centred at
`T(fit_M) − T(fit_R)` from `moment_fit` (zero when either fit is unavailable).
"""
struct FitStudentT{S} <: ContrastProposal
    ν::Float64
    s::S
end

"""
    ZeroStudentT(ν, s)

Per-coordinate Student-t with `ν` degrees of freedom and scale `s`, centred at zero.
"""
struct ZeroStudentT{S} <: ContrastProposal
    ν::Float64
    s::S
end

"""
    FitNormal(s)

Per-coordinate Normal with scale `s`, centred as in `FitStudentT`.
"""
struct FitNormal{S} <: ContrastProposal
    s::S
end

"""
    Coupled(contrast, p)

A coupled-parameter birth/death, used on a birth or death with probability `p`
(the original move otherwise). A birth splits the original cluster `O` into
the remainder `R` and the moving set `M` with `t_R = t_O − (|M|/|O|) u` and
`t_M = t_O + (|R|/|O|) u`, `u ~ g`; a death merges `M` into `B` with the
inverse map. The map has unit Jacobian on the unconstrained scale, and each
cluster factor carries its own `|det DT⁻¹|`.
"""
struct Coupled{G<:ContrastProposal}
    contrast::G
    p::Float64
    function Coupled(contrast::G, p::Real = 0.5) where {G<:ContrastProposal}
        0 < p <= 1 || throw(ArgumentError("p must be in (0, 1]"))
        new{G}(contrast, float(p))
    end
end

"""
    CoupledRJ(proposal, coupled::Coupled)

Birth proposal for the reversible jump update that adds the coupled birth/death
of `coupled` to the original birth/death with parameter proposal `proposal`.
"""
struct CoupledRJ{P<:BirthProposal, C<:Coupled} <: BirthProposal
    proposal::P
    coupled::C
end

"""
    AuxiliaryMH{P<:BirthProposal} <: BirthProposal

Select the Metropolised form of Algorithm 8 for the links: a reversible jump
update in which `c_i` is proposed by the link proposal and accepted by
Metropolis-Hastings, and a move that creates or removes the moving set's own
cluster weighs `m` auxiliary parameters from `proposal`, the current value
taking a random slot when it exists. With `m = 1` it is the reversible jump
update with `NoUpdate()`, as a birth weighs one draw from `q`. Larger `m`
lowers the variance of the birth and death ratios at `m` likelihood
evaluations of the moving set, paid only on those moves. A rejected death
refreshes the moving set's parameter from the pool. The fixed-dimension
proposal passed to `mcmc` is ignored (clusters keep their parameters).

# Fields
- `proposal::P`: Birth density `q(· | y_M)`
- `m::Int`: Number of auxiliary parameters
- `coupled`: `nothing`, or a `Coupled` move mixed into births and deaths (keyword)
"""
struct AuxiliaryMH{P<:BirthProposal, C<:Union{Nothing, Coupled}} <: BirthProposal
    proposal::P
    m::Int
    coupled::C
    function AuxiliaryMH(proposal::P, m::Int; coupled::C = nothing) where {P<:BirthProposal, C<:Union{Nothing, Coupled}}
        m >= 1 || throw(ArgumentError("m must be at least 1"))
        proposal isa Union{AuxiliaryGibbs, AuxiliaryMH, CoupledRJ} && throw(ArgumentError("nested auxiliary proposal"))
        new{P, C}(proposal, m, coupled)
    end
end

# ============================================================================
# Two-anchor multi-link split/merge
# ============================================================================

"""
    TwoAnchor(; k = 3.0, attempts = 1, pilot = false, tilt = 1.0, p_cross = 0.9)

An occasional multi-link split/merge (pass as `MCMCOptions(two_anchor = ...)`):
`attempts` times per sweep an ordered pair of distinct anchors `(a, b)` is drawn
uniformly; `U` is the union of their clusters. Every `i ∈ U` redraws its link
with probability `p = min(0.5, k/|U|)` from the ddCRP weights restricted to `U`,
and the proposal is a split (anchors together) if it leaves exactly two
components with `a` and `b` apart, or a merge (anchors apart) if it leaves one;
any other topology is a rejection. The cluster holding `a` keeps its parameter;
a split draws the other's from the birth proposal. Exact for any base update
with explicit cluster parameters.

With `pilot = true` the links are guided by a deterministic two-way grouping of
`U` built from the anchors and the data only (`two_anchor_pilot`), so it is the
same in both directions. Each `i` redraws its link with probability
`max(p, p_cross · s_i)`, where `s_i` is the share of its link weight in the
other pilot group, so the redraws concentrate on the pilot boundary. With
`tilt = 1` (the default) targets are the ddCRP weights and the link terms of
the ratio cancel; `tilt > 1` favours same-group targets on a split and
other-group targets on a merge. Every target in `U` keeps positive probability.
"""
struct TwoAnchor
    k::Float64
    attempts::Int
    pilot::Bool
    tilt::Float64
    p_cross::Float64
    function TwoAnchor(; k::Real = 3.0, attempts::Int = 1, pilot::Bool = false, tilt::Real = 1.0,
                       p_cross::Real = 0.9)
        k > 0 || throw(ArgumentError("k must be positive"))
        attempts >= 1 || throw(ArgumentError("attempts must be at least 1"))
        tilt > 0 || throw(ArgumentError("tilt must be positive"))
        0 < p_cross < 1 || throw(ArgumentError("p_cross must be in (0, 1)"))
        new(float(k), attempts, pilot, float(tilt), float(p_cross))
    end
end

# ============================================================================
# Fixed-Dimension Proposals for RJMCMC
# ============================================================================

"""
    FixedDimensionProposal

Abstract supertype for RJMCMC fixed-dimension proposal distributions.
Controls how cluster parameters are updated when the moving set S_i transfers
between existing clusters without changing the total number of clusters K.
"""
abstract type FixedDimensionProposal end

"""
    NoUpdate <: FixedDimensionProposal

Keep existing cluster parameters unchanged during fixed-dimension moves.
The acceptance probability depends solely on the posterior ratio.
"""
struct NoUpdate <: FixedDimensionProposal end

"""
    WeightedMean <: FixedDimensionProposal

Deterministically update parameters as weighted averages of cluster contents.
For a parameter ρ, the augmented cluster gets a weighted mean incorporating
the moving set, and the depleted cluster is adjusted accordingly.
The update is deterministic (lpr = 0) so the Jacobian is unity.
"""
struct WeightedMean <: FixedDimensionProposal end

"""
    Resample{P<:BirthProposal} <: FixedDimensionProposal

Stochastically resample cluster parameters for the modified clusters using
an inner `BirthProposal`. Reuses `sample_birth_param`/`birth_param_logpdf`
applied to the new cluster memberships (remaining depleted, augmented).
The Hastings ratio accounts for the forward and reverse proposal densities.

# Fields
- `proposal::P`: The birth proposal to use for resampling

# Example
```julia
Resample(NormalMomentMatch(0.5, 0.3, 0.5))  # moment-matched resampling
Resample()                                    # prior-based resampling
```
"""
struct Resample{P<:BirthProposal} <: FixedDimensionProposal
    proposal::P
end
Resample() = Resample(PriorProposal())

"""
    MixedFixedDim{T<:NamedTuple} <: FixedDimensionProposal

Compose per-parameter fixed-dimension proposals. Each cluster parameter can use
a different update strategy. The `proposals` field is a NamedTuple mapping
parameter names to individual `FixedDimensionProposal` instances.
Unspecified parameters default to `NoUpdate`.

# Example
```julia
MixedFixedDim(ξ = WeightedMean(), ω = NoUpdate(), α = NoUpdate())
```
"""
struct MixedFixedDim{T<:NamedTuple} <: FixedDimensionProposal
    proposals::T
end
MixedFixedDim(; kwargs...) = MixedFixedDim(NamedTuple(kwargs))


# ============================================================================
# Observed Data Types
# ============================================================================

"""
    AbstractObservedData

Abstract supertype for observed data containers in the DDCRP framework.
Encapsulates response data (y), distance matrix (D), and optional trials data.
"""
abstract type AbstractObservedData end

"""
    CountData{Ty, Td} <: AbstractObservedData

Observed count data for Poisson and Binomial models.

# Fields
- `y::Ty`: Observed counts (AbstractVector)
- `D::Td`: Distance matrix (AbstractMatrix)
"""
struct CountData{Ty<:AbstractVector, Td<:AbstractMatrix} <: AbstractObservedData
    y::Ty
    D::Td
end

"""
    CountDataWithTrials{Ty, Tn, Td} <: AbstractObservedData

Observed count data with number of trials for Binomial models.

# Fields
- `y::Ty`: Observed successes (AbstractVector)
- `N::Tn`: Number of trials (scalar Int or AbstractVector{Int})
- `D::Td`: Distance matrix (AbstractMatrix)
"""
struct CountDataWithTrials{Ty<:AbstractVector, Tn<:Union{Int, <:AbstractVector{Int}}, Td<:AbstractMatrix} <: AbstractObservedData
    y::Ty
    N::Tn
    D::Td
end

"""
    CountDataWithPopulation{Ty, Tp, Td} <: AbstractObservedData

Observed count data with population/exposure offsets for Poisson/NB population models.

# Fields
- `y::Ty`: Observed counts (AbstractVector)
- `P::Tp`: Population or exposure (scalar or AbstractVector{<:Real})
- `D::Td`: Distance matrix (AbstractMatrix)
- `missing_mask::BitVector`: `true` for indices with missing observations (default: all false)
"""
struct CountDataWithPopulation{Ty<:AbstractVector, Tp<:Union{<:Real,<:AbstractVector{<:Real}}, Td<:AbstractMatrix} <: AbstractObservedData
    y::Ty
    P::Tp
    D::Td
    missing_mask::BitVector
end
CountDataWithPopulation(y, P, D) = CountDataWithPopulation(y, P, D, falses(length(y)))

"""
    ContinuousData{Ty, Td} <: AbstractObservedData

Observed continuous data for continuous-valued models (e.g. Gamma).

# Fields
- `y::Ty`: Observed values (AbstractVector{<:Real})
- `D::Td`: Distance matrix (AbstractMatrix)
"""
struct ContinuousData{Ty<:AbstractVector{<:Real}, Td<:AbstractMatrix} <: AbstractObservedData
    y::Ty
    D::Td
end

# ============================================================================
# Observed Data Accessor Functions
# ============================================================================

"""Return the observations vector."""
observations(data::AbstractObservedData) = data.y

"""Return the distance matrix."""
distance_matrix(data::AbstractObservedData) = data.D

"""Return the number of trials (only for CountDataWithTrials)."""
trials(data::CountDataWithTrials) = data.N

"""Check if data has trials information."""
has_trials(::AbstractObservedData) = false
has_trials(::CountDataWithTrials) = true

"""Return the population/exposure vector (only for CountDataWithPopulation)."""
population(data::CountDataWithPopulation) = data.P

"""Check if data has population/exposure information."""
has_population(::AbstractObservedData) = false
has_population(::CountDataWithPopulation) = true

"""Return the missing data mask (BitVector of length n); nothing for other data types."""
get_missing_mask(data::CountDataWithPopulation) = data.missing_mask
get_missing_mask(::AbstractObservedData) = nothing

"""Return true if any observations are missing."""
has_missing(data::CountDataWithPopulation) = any(data.missing_mask)
has_missing(::AbstractObservedData) = false

"""Number of observations."""
nobs(data::AbstractObservedData) = length(data.y)

# ============================================================================
# Model Data Requirement Traits
# ============================================================================

"""
    requires_trials(model::LikelihoodModel) -> Bool

Returns true if the model requires data with trials/exposure (N or P).
"""
requires_trials(::LikelihoodModel) = false
requires_trials(::BinomialModel) = true

"""
    requires_population(model::LikelihoodModel) -> Bool

Returns true if the model requires data with population/exposure offsets (P).
"""
requires_population(::LikelihoodModel) = false




 # Types for missing data handling when updating c
abstract type CustomerUpdate end

struct StandardUpdate <: CustomerUpdate end
struct MissingUpdate <: CustomerUpdate end

# ============================================================================
# Link proposals
#
# How a new value of c_i is proposed in the RJMCMC assignment update. The
# choice matters most when the decay function gives many pairs negligible
# weight (a truncated or windowed kernel, say): a uniform draw then spends most
# proposals on links the prior all but forbids, each of which still costs a
# likelihood evaluation before it is rejected.
# ============================================================================

abstract type LinkProposal end

"""
    UniformLink()

Propose `c_i` uniformly over the `n` observations. Symmetric, so the proposal
ratio is 1 and the ddCRP prior ratio remains in the acceptance probability.
"""
struct UniformLink <: LinkProposal end

"""
    PriorLink()

Propose `c_i` from the ddCRP prior, `Pr(c_i = j) ∝ f(d_ij)` for `j ≠ i` and
`∝ α` for `j = i`. The proposal ratio then cancels the prior ratio exactly, so
the acceptance probability depends only on the likelihood ratio and the
parameter proposal, and every proposal lands where the prior has mass.
"""
struct PriorLink <: LinkProposal end

"""
    MetropolisedPriorLink()

Propose `c_i` from the ddCRP prior restricted to links that change the
partition, i.e. outside the block of `z(c_{-i})` holding the current link,
and accept with the ratio `(1 - P(b_old)) / (1 - P(b_new))` of the prior
masses outside each block (Liu, 1996). Each accepted or rejected move is
followed by an exact draw of `c_i` within its block, which leaves the
partition and likelihood unchanged. Every move off the diagonal is at least as
likely as under `PriorLink`, so it is never worse in the Peskun order.
Supported by `AuxiliaryMH` only.
"""
struct MetropolisedPriorLink <: LinkProposal end

"""
    InformedLink(ε = 0.1, τ = 1.0)

Propose `c_i` by first choosing a block `b` of `z(c_{-i})` with probability
`(1−ε) softmax(τ s)_b + ε W_b / Σ W`, then a link within it in proportion to
its ddCRP weight. The score of an existing block is `log W_b + log p(y_M | ρ_b)`;
the moving set's own block is scored with a deterministic plug-in
`log W_M + table_contribution(M; ρ̂) − (d/2) log |M|` at the `moment_fit` (or
prior-mean) parameters. Block parameters are unchanged by the move, so the
forward and reverse proposals share one score vector. Requires `NoUpdate()`
and no `Coupled` move.
"""
struct InformedLink <: LinkProposal
    ε::Float64
    τ::Float64
    function InformedLink(ε::Real = 0.1, τ::Real = 1.0)
        0 < ε <= 1 || throw(ArgumentError("ε must be in (0, 1]"))
        new(float(ε), float(τ))
    end
end
