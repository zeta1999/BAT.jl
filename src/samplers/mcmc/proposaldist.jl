# This file is a part of BAT.jl, licensed under the MIT License (MIT).


@doc doc"""
    AbstractProposalDist

*BAT-internal, not part of stable public API.*

The following functions must be implemented for subtypes:

* `BAT.distribution_logpdf`
* `BAT.proposal_rand!`
* `ValueShapes.totalndof`, returning the number of DOF (i.e. dimensionality).
* `LinearAlgebra.issymmetric`, indicating whether p(a -> b) == p(b -> a) holds true.

In some cases, it may be desirable to override the default implementation
of `BAT.distribution_logpdf!`.
"""
abstract type AbstractProposalDist end


@doc doc"""
    distribution_logpdf(
        p::AbstractArray,
        pdist::AbstractProposalDist,
        v_proposed::AbstractVector,
        v_current:::AbstractVector
    )

*BAT-internal, not part of stable public API.*

Analog to `distribution_logpdf!`, but for a single variate/parameter vector.
"""
function distribution_logpdf end

# TODO: Implement distribution_logpdf for included proposal distributions


@doc doc"""
    distribution_logpdf!(
        p::AbstractArray,
        pdist::AbstractProposalDist,
        v_proposed::Union{AbstractVector,VectorOfSimilarVectors},
        v_current:::Union{AbstractVector,VectorOfSimilarVectors}
    )

*BAT-internal, not part of stable public API.*

Returns log(PDF) value of `pdist` for transitioning from current to proposed
variate/parameter arguments values for a variate/parameter sets.

end

Input:

* `v_proposed`: New values (column vectors)
* `v_current`: Old values (column vectors)

Output is stored in

* `p`: Array of PDF values, length must match, shape is ignored

Array size requirements:

* `size(v_current, 1) == size(v_proposed, 1) == length(pdist)`
* `size(v_current, 2) == size(v_proposed, 2)` or `size(v_current, 2) == 1`
* `size(v_proposed, 2) == length(p)`

Implementations of `distribution_logpdf!` must be thread-safe.
"""
function distribution_logpdf! end

# TODO: Default implementation of distribution_logpdf!


@doc doc"""
    function proposal_rand!(
        rng::AbstractRNG,
        pdist::GenericProposalDist,
        v_proposed::Union{AbstractVector,VectorOfSimilarVectors},
        v_current::Union{AbstractVector,VectorOfSimilarVectors}
    )

*BAT-internal, not part of stable public API.*

Generate one or multiple proposed variate/parameter vectors, based on one or
multiple previous vectors.

Input:

* `rng`: Random number generator to use
* `pdist`: Proposal distribution to use
* `v_current`: Old values (vector or column vectors, if a matrix)

Output is stored in

* `v_proposed`: New values (vector or column vectors, if a matrix)

The caller must guarantee:

* `size(v_current, 1) == size(v_proposed, 1)`
* `size(v_current, 2) == size(v_proposed, 2)` or `size(v_current, 2) == 1`
* `v_proposed !== v_current` (no aliasing)

Implementations of `proposal_rand!` must be thread-safe.
"""
function proposal_rand! end



struct GenericProposalDist{D<:Distribution{Multivariate},SamplerF,S<:Sampleable} <: AbstractProposalDist
    d::D
    sampler_f::SamplerF
    s::S

    function GenericProposalDist{D,SamplerF}(d::D, sampler_f::SamplerF) where {D<:Distribution{Multivariate},SamplerF}
        s = sampler_f(d)
        new{D,SamplerF, typeof(s)}(d, sampler_f, s)
    end

end


GenericProposalDist(d::D, sampler_f::SamplerF) where {D<:Distribution{Multivariate},SamplerF} =
    GenericProposalDist{D,SamplerF}(d, sampler_f)

GenericProposalDist(d::Distribution{Multivariate}) = GenericProposalDist(d, bat_sampler)

GenericProposalDist(D::Type{<:Distribution{Multivariate}}, varndof::Integer, args...) =
    GenericProposalDist(D, Float64, varndof, args...)


Base.similar(q::GenericProposalDist, d::Distribution{Multivariate}) =
    GenericProposalDist(d, q.sampler_f)

function Base.convert(::Type{AbstractProposalDist}, q::GenericProposalDist, T::Type{<:AbstractFloat}, varndof::Integer)
    varndof != totalndof(q) && throw(ArgumentError("q has wrong number of DOF"))
    q
end


get_cov(q::GenericProposalDist) = get_cov(q.d)
set_cov(q::GenericProposalDist, Σ::PosDefMatLike) = similar(q, set_cov(q.d, Σ))


function distribution_logpdf!(
    p::AbstractArray,
    pdist::GenericProposalDist,
    v_proposed::VectorOfSimilarVectors,
    v_current::Union{AbstractVector,VectorOfSimilarVectors}
)
    params_diff = flatview(v_proposed) .- flatview(v_current) # TODO: Avoid memory allocation
    Distributions.logpdf!(p, pdist.d, params_diff)
end


function distribution_logpdf!(
    p::AbstractArray,
    pdist::GenericProposalDist,
    v_proposed::AbstractVector,
    v_current::AbstractVector
)
    p[1] = distribution_logpdf(pdist, v_proposed, v_current)
    p
end


function distribution_logpdf(
    pdist::GenericProposalDist,
    v_proposed::AbstractVector,
    v_current::AbstractVector
)
    params_diff = v_proposed .- v_current # TODO: Avoid memory allocation
    Distributions.logpdf(pdist.d, params_diff)
end


function proposal_rand!(
    rng::AbstractRNG,
    pdist::GenericProposalDist,
    v_proposed::Union{AbstractVector,VectorOfSimilarVectors},
    v_current::Union{AbstractVector,VectorOfSimilarVectors}
)
    rand!(rng, pdist.s, flatview(v_proposed))
    params_new_flat = flatview(v_proposed)
    params_new_flat .+= flatview(v_current)
    v_proposed
end


ValueShapes.totalndof(pdist::GenericProposalDist) = length(pdist.d)

LinearAlgebra.issymmetric(pdist::GenericProposalDist) = issymmetric_around_origin(pdist.d)



struct GenericUvProposalDist{D<:Distribution{Univariate},T<:Real,SamplerF,S<:Sampleable} <: AbstractProposalDist
    d::D
    scale::Vector{T}
    sampler_f::SamplerF
    s::S
end


GenericUvProposalDist(d::Distribution{Univariate}, scale::Vector{<:AbstractFloat}, samplerF) =
    GenericUvProposalDist(d, scale, samplerF, samplerF(d))

GenericUvProposalDist(d::Distribution{Univariate}, scale::Vector{<:AbstractFloat}) =
    GenericUvProposalDist(d, scale, bat_sampler)


ValueShapes.totalndof(pdist::GenericUvProposalDist) = size(pdist.scale, 1)

LinearAlgebra.issymmetric(pdist::GenericUvProposalDist) = issymmetric_around_origin(pdist.d)

function BAT.distribution_logpdf(
    pdist::GenericUvProposalDist,
    v_proposed::Union{AbstractVector,VectorOfSimilarVectors},
    v_current::Union{AbstractVector,VectorOfSimilarVectors}
)
    params_diff = (flatview(v_proposed) .- flatview(v_current)) ./ pdist.scale  # TODO: Avoid memory allocation
    sum_first_dim(Distributions.logpdf.(pdist.d, params_diff))  # TODO: Avoid memory allocation
end

function BAT.proposal_rand!(
    rng::AbstractRNG,
    pdist::GenericUvProposalDist,
    v_proposed::AbstractVector,
    v_current::AbstractVector
)
    v_proposed .= v_current
    dim = rand(rng, eachindex(pdist.scale))
    v_proposed[dim] += pdist.scale[dim] * rand(rng, pdist.s)
    v_proposed
end



abstract type ProposalDistSpec end


struct MvTDistProposal <: ProposalDistSpec
    df::Float64
end

MvTDistProposal() = MvTDistProposal(1.0)


(ps::MvTDistProposal)(T::Type{<:AbstractFloat}, varndof::Integer) =
    GenericProposalDist(MvTDist, T, varndof, convert(T, ps.df))

function GenericProposalDist(::Type{MvTDist}, T::Type{<:AbstractFloat}, varndof::Integer, df = one(T))
    Σ = PDMat(Matrix(ScalMat(varndof, one(T))))
    μ = Fill(zero(eltype(Σ)), varndof)
    M = typeof(Σ)
    d = Distributions.GenericMvTDist(convert(T, df), μ, Σ)
    GenericProposalDist(d)
end


struct UvTDistProposalSpec <: ProposalDistSpec
    df::Float64
end

(ps::UvTDistProposalSpec)(T::Type{<:AbstractFloat}, varndof::Integer) =
    GenericUvProposalDist(TDist(convert(T, ps.df)), fill(one(T), varndof))
