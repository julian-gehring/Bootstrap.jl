abstract type Model end

struct SimpleModel{T,A,K} <: Model
    class::T
    args::A
    kwargs::K
end

Model(class, args...; kwargs...) = SimpleModel(class, args, kwargs)

struct FormulaModel{T,A,K} <: Model
    class::T
    formula::FormulaTerm
    args::A
    kwargs::K
end

Model(class, formula::FormulaTerm, args...; kwargs...) =
    FormulaModel(class, formula, args, kwargs)

abstract type BootstrapSampling end

abstract type ParametricSampling <: BootstrapSampling end
abstract type NonParametricSampling <: BootstrapSampling end


"""
Basic Sampling

```julia
BasicSampling(1000)
```

"""
struct BasicSampling{R<:AbstractRNG} <: BootstrapSampling
    rng::R
    nrun::Int
end

BasicSampling(nrun::Int) = BasicSampling(Random.GLOBAL_RNG, nrun)

"""
Antithetic Sampling

```julia
AntitheticSampling(1000)
```

"""
struct AntitheticSampling{R<:AbstractRNG} <: BootstrapSampling
    rng::R
    nrun::Int
end

AntitheticSampling(nrun::Int) = AntitheticSampling(Random.GLOBAL_RNG, nrun)

"""
Balanced Sampling

```julia
BalancedSampling(1000)
```

"""
struct BalancedSampling{R<:AbstractRNG} <: BootstrapSampling
    rng::R
    nrun::Int
end

BalancedSampling(nrun::Int) = BalancedSampling(Random.GLOBAL_RNG, nrun)

"""
Residual Sampling

```julia
ResidualSampling(1000)
```

"""
struct ResidualSampling{R<:AbstractRNG} <: BootstrapSampling
    rng::R
    nrun::Int
end

ResidualSampling(nrun::Int) = ResidualSampling(Random.GLOBAL_RNG, nrun)

"""
Wild Sampling

```julia
WildSampling(1000, rademacher)
WildSampling(1000, mammen)
```

"""
struct WildSampling{R<:AbstractRNG,F} <: BootstrapSampling
    rng::R
    nrun::Int
    noise::F
end

WildSampling(nrun::Int, noise) = WildSampling(Random.GLOBAL_RNG, nrun, noise)

"""
Exact Sampling

```julia
ExactSampling()
```

"""
mutable struct ExactSampling <: BootstrapSampling
    nrun::Int
end

ExactSampling() = ExactSampling(0)

"""
[Maximum Entropy Sampling](https://cran.r-project.org/web/packages/meboot/vignettes/meboot.pdf)

```julia
MaximumEntropySampling(100)
maximumEntropySampling(100, MaximumEntropyCache())
```

NOTE: Implementation based off [pymeboot](https://github.com/kirajcg/pymeboot) as the original
[R package](https://cran.r-project.org/web/packages/meboot/index.html) is GPL licensed.
"""
struct MaximumEntropySampling{R,T} <: BootstrapSampling
    rng::R
    nrun::Int
    cache::MaximumEntropyCache{T}
end

MaximumEntropySampling(nrun::Int) = MaximumEntropySampling(Random.GLOBAL_RNG, nrun)
MaximumEntropySampling(rng, nrun::Int) =
    MaximumEntropySampling(rng, nrun, MaximumEntropyCache())

abstract type BootstrapSample end

struct NonParametricBootstrapSample{T,T0,T1,S,B} <: BootstrapSample
    t0::T0
    t1::T1
    statistic::S
    data::T
    sampling::B
end

struct ParametricBootstrapSample{T,M,T0,T1,S,B} <: BootstrapSample
    t0::T0
    t1::T1
    statistic::S
    data::T
    model::M
    sampling::B
end


"""
Number of samples drawn from a bootstrap sampling

```julia
bs = BasicSampling(1000)
nrun(bs)

# output

1000
```

"""
nrun(bs::BootstrapSampling) = bs.nrun
nrun(bs::BootstrapSample) = nrun(sampling(bs))


"""
Return the statistic function of a `BootstrapSample`

```julia
bs = bootstrap(randn(20), mean, BasicSampling(100))
statistic(bs)
```

"""
statistic(bs::BootstrapSample) = bs.statistic

"""
Return the noise function of a wild bootstrap sampling
"""
noise(bs::WildSampling) = bs.noise

nvar(t::Tuple) = length(t)
nvar(bs::BootstrapSample) = nvar(original(bs))

tx(x) = tuple(x...)

zeros_tuple(t::Tuple, m::Int) = _zeros_tuple(typeof(t), m)
_zeros_tuple(::Type{Tuple{}}, m::Int) = ()
_zeros_tuple(::Type{T}, m::Int) where T <: Tuple =
    (zeros(Base.tuple_type_head(T), m), _zeros_tuple(Base.tuple_type_tail(T), m)...)

"""
    bootstrap(statistic, data[, model], sampling)

Bootstrap the `statistic` of dataset `data` using the resampling method `sampling`.

If a `model` is specified, a parametric bootstrap is performed.
"""
function bootstrap end

function bootstrap(statistic, data, sampling::BasicSampling)
    t0 = tx(statistic(data))
    m = nrun(sampling)
    t1 = zeros_tuple(t0, m)
    data1 = copy(data)
    rng = sampling.rng
    for i in 1:m
        draw!(rng, data, data1)
        for (j, t) in enumerate(tx(statistic(data1)))
            t1[j][i] = t
        end
    end
    return NonParametricBootstrapSample(t0, t1, statistic, data, sampling)
end


function bootstrap(statistic, data::AbstractVector, sampling::AntitheticSampling)
    t0 = tx(statistic(data))
    m = nrun(sampling)
    n = nobs(data)
    t1 = zeros_tuple(t0, m)
    idx = collect(1:n)
    idx1 = copy(idx)
    data1 = copy(data)
    data0 = copy(data)
    sort!(data0)
    rng = sampling.rng
    for i in 1:m
        if isodd(i)
            sample!(rng, idx, idx1)
        else
            idx1 = n .- idx1 .+ 1
        end
        data1 = pick(data0, idx1)
        for (j, t) in enumerate(tx(statistic(data1)))
            t1[j][i] = t
        end
    end
    return NonParametricBootstrapSample(t0, t1, statistic, data, sampling)
end

function bootstrap(statistic, data, sampling::BalancedSampling)
    n = nobs(data)
    m = nrun(sampling)
    t0 = tx(statistic(data))
    t1 = zeros_tuple(t0, m)
    idx = repeat([1:n;], m)
    ridx = zeros(Int, n, m)
    sample!(sampling.rng, idx, ridx, replace = false)
    for i in 1:m
        for (j, t) in enumerate(tx(statistic(pick(data, ridx[:,i]))))
            t1[j][i] = t
        end
    end
    return NonParametricBootstrapSample(t0, t1, statistic, data, sampling)
end

struct ExactIterator{T}
    a::T
    k::Int
end

exact(n::Int) = ExactIterator(1:n, n)

function iterate(itr::ExactIterator, s = ones(Int, itr.k))
    if length(s) > 0 && s[1] < 1
        return nothing
    end
    r = itr.a[s]
    for i = itr.k:-1:1
        if s[i] < lastindex(itr.a)
            s[i] = nextind(itr.a, s[i])
            for j = i:itr.k - 1
                s[j + 1] = s[j]
            end
            return (r, s)
        end
    end
    s[1] = 0
    return (r, s)
end

eltype(itr::ExactIterator) = typeof(itr.a)
eltype(itr::ExactIterator{UnitRange{T}}) where {T} = Array{T,1}
eltype(itr::ExactIterator{AbstractRange{T}}) where {T} = Array{T,1}

length(itr::ExactIterator) = binomial(length(itr.a) + itr.k - 1, itr.k)
size(itr::ExactIterator) = (itr.k,)


function bootstrap(statistic, data, sampling::ExactSampling)
    n = nobs(data)
    m = nrun_exact(n)
    t0 = tx(statistic(data))
    t1 = zeros_tuple(t0, m)
    for (i, idx) in enumerate(exact(n))
        for (j, t) in enumerate(tx(statistic(pick(data, idx))))
            t1[j][i] = t
        end
    end
    sampling.nrun = m
    return NonParametricBootstrapSample(t0, t1, statistic, data, sampling)
end

function bootstrap(statistic, data, sampling::MaximumEntropySampling)
    cache = sampling.cache
    init!(cache, data)

    t0 = tx(statistic(data))
    m = nrun(sampling)
    t1 = zeros_tuple(t0, m)
    data1 = copy(data)
    rng = sampling.rng
    for i in 1:m
        draw!(rng, cache, data, data1)
        for (j, t) in enumerate(tx(statistic(data1)))
            t1[j][i] = t
        end
    end
    return NonParametricBootstrapSample(t0, t1, statistic, data, sampling)
end

function bootstrap(statistic, data, model::SimpleModel, sampling::BootstrapSampling)
    f0 = fit(model.class, data, model.args...; model.kwargs...)
    t0 = tx(statistic(data))
    m = nrun(sampling)
    t1 = zeros_tuple(t0, m)
    data1 = copy(data)
    rng = sampling.rng
    for i in 1:m
        draw!(rng, f0, data1)
        for (j, t) in enumerate(tx(statistic(data1)))
            t1[j][i] = t
        end
    end
    return ParametricBootstrapSample(t0, t1, statistic, data, model, sampling)
end

struct ResidualTerm{S,T,R} <: StatsModels.AbstractTerm
    sampling::S
    t::T
    r0::R
    r1::R
    ResidualTerm(s::S, t::T, r::R) where {S,T,R} = new{S,T,R}(s, t, r, copy(r))
end


StatsModels.modelcols(t::ResidualTerm, data) =
    throw(ArgumentError("ResidualTerm: don't know how to generate model columns for $(typeof(t.sampling))"))
StatsModels.modelcols(t::ResidualTerm{<:ResidualSampling}, data) =
    modelcols(t.t, data) + sample!(t.sampling.rng, t.r0, t.r1)
StatsModels.modelcols(t::ResidualTerm{<:WildSampling}, data) =
    modelcols(t.t, data) + t.sampling.noise(t.sampling.rng, t.r0)

# we only ever create one of these terms after the model has been fit once so we don't
# need to create a new schema for it every time...
StatsModels.needs_schema(t::ResidualTerm) = false
StatsModels.termvars(t::ResidualTerm) = StatsModels.termvars(t.t)

function bootstrap(statistic, data::AbstractDataFrame, model::FormulaModel,
                   sampling::BootstrapSampling)
    class = model.class
    formula = apply_schema(model.formula, schema(model.formula, data), model.class)

    args = model.args
    kwargs = model.kwargs

    f0 = fit(class, formula, data, args...; kwargs...)
    r0 = predict(f0) - response(f0)
    # replace LHS of formula with a special term that resamples from the residuals
    # whenever it's asked for model columns
    formula_resid = FormulaTerm(ResidualTerm(sampling, formula.lhs, r0), formula.rhs)

    m = nrun(sampling)
    t0 = tx(statistic(f0))
    t1 = zeros_tuple(t0, m)

    for i in 1:m
        f1 = fit(class, formula_resid, data, args...; kwargs...)
        for (j, t) in enumerate(tx(statistic(f1)))
            t1[j][i] = t
        end
    end
    return ParametricBootstrapSample(t0, t1, statistic, data, model, sampling)
end
