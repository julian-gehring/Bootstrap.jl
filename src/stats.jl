## rademacher
rademacher(rng::AbstractRNG, x) = x .* sign.(randn(rng, nobs(x)))

const _mammen_dist = Binomial(1, (sqrt(5) + 1) / (2 * sqrt(5)))
const _mammen_val1 = -(sqrt(5) - 1) / 2
const _mammen_val2 = (sqrt(5) + 1) / 2

function mammen(rng::AbstractRNG, x)
    r = rand(rng, _mammen_dist, nobs(x))
    return ifelse.(r .== 1, _mammen_val1, _mammen_val2)
end

## Number of exact bootstrap runs
nrun_exact(n::Integer) = binomial(2n - 1, n)

## Jack-Knife estimate
function jack_knife_estimate(data, statistic::Function, j::Int = 1,
                             ::Type{T} = typeof(statistic(data)[j])) where T
    n = nobs(data)
    y = zeros(T, n)
    idx = trues(n)
    for i in 1:n
        idx[i] = false
        if i > 1
            idx[i - 1] = true
        end
        y[i] = tx(statistic(pick(data, idx)))[j]
    end
    return y
end


## Sample quantiles with Gaussian interpolation
## Davison and Hinkley, equation 5.6

function iquantile(x::AbstractVector, alpha::AbstractVector)
    x = sort(x)
    qn = Float64[iquantile(x, a, true) for a in alpha]
    return qn
end


function iquantile(x::AbstractVector, alpha::AbstractFloat, is_sorted::Bool = false)
    if !is_sorted
        x = sort(x)
    end
    n = length(x)
    k = trunc(Int, (n + 1) * alpha)
    ## infinity outside of data range
    if k == 0
        return -Inf
    elseif k > (n - 1)
        return Inf
    end
    tx = [quantile(Normal(), a) for a in [alpha, k / (n + 1), (k + 1) / (n + 1)]]
    qn = (tx[1] - tx[2]) / (tx[3] - tx[2]) * (x[k + 1] - x[k]) + x[k]
    return qn
end
