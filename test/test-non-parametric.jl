module TestNonParametricBootstrap

using Bootstrap
using Test

using Statistics
using Random

using DataFrames
using RDatasets
using StatsBase

## 'city' dataset
const city = dataset("boot", "city")
const citya = Matrix(city)

@testset "Non-parametric bootstraps" begin

    function test_bootsample(bs, ref, raw_data, n)

        show(IOBuffer(), bs)
        @test typeof(bs) <: NonParametricBootstrapSample
        t0 = original(bs)
        @test length(t0) == length(ref)
        [@test t ≈ r for (t, r) in zip(t0, ref)]

        t1 = straps(bs)
        @test length(t1) == length(t0)
        [@test length(t) == n for t in t1]
        [@test (minimum(t) <= tr && maximum(t) >= tr) for (t, tr) in zip(t1, t0)]
        [@test eltype(t) == eltype(tr) for (t, tr) in zip(t1, t0)]

        @test Bootstrap.data(bs) == raw_data

        @test nrun(sampling(bs)) == n
        @test nrun(bs) == n

        @test length(bias(bs)) == length(ref)
        [@test (b[1] > -Inf && b[1] < Inf) for b in bias(bs)]
        @test length(stderror(bs)) == length(ref)
        [@test s >= 0 for s in stderror(bs)]

        [@test original(bs, i) == original(bs)[i]  for i in 1:nvar(bs)]
        [@test straps(bs, i) == straps(bs)[i]  for i in 1:nvar(bs)]
        [@test bias(bs, i) == bias(bs)[i]  for i in 1:nvar(bs)]
        [@test stderror(bs, i) == stderror(bs)[i]  for i in 1:nvar(bs)]

        @test_throws MethodError model(bs)

        return Nothing
    end

    function test_confint(bs)

        cim_all = (BasicConfInt(), PercentileConfInt(), NormalConfInt(), BCaConfInt())
        for cim in cim_all
            c = confint(bs, cim)
            [@test (x[1] >= x[2]  && x[1] <= x[3]) for x in c]
            [@test x[1] ≈ t0 for (x, t0) in zip(c, original(bs))]
            @test level(cim) == 0.95
        end

        return Nothing
    end

    function test_confint_width0(bs)

        cim_all = (BasicConfInt(), PercentileConfInt(), NormalConfInt(), BCaConfInt())
        for cim in cim_all
            c = confint(bs, cim)
            # We expect a confidence interval of width 0
            [@test x[2] == x[1] == x[3] for x in c]
            [@test x[1] ≈ t0 for (x, t0) in zip(c, original(bs))]
            @test level(cim) == 0.95
        end

        return Nothing
    end

    n = 250

    city_ratio(x::AbstractArray) = mean(x[:,2]) ./ mean(x[:,1])
    city_ratio(x::AbstractDataFrame) = mean(x[!,:X]) ./ mean(x[!,:U])

    city_cor(x::AbstractArray) = cor(x[:,1], x[:,2])
    city_cor(x::AbstractDataFrame) = cor(x[!,:X], x[!,:U])


    @testset "Basic resampling" begin

        @testset "city_ratio with DataFrame input" begin
            ref = city_ratio(city)
            @test ref  ≈ 1.5203125
            bs = bootstrap(city_ratio, city, BasicSampling(n))
            test_bootsample(bs, ref, city, n)
            test_confint(bs)
        end

        @testset "city_cor with DataFrame input" begin
            ref = city_cor(city)
            bs = bootstrap(city_cor, city, BasicSampling(n))
            test_bootsample(bs, ref, city, n)
            test_confint(bs)
        end

        @testset "city_cor with DataArray input" begin
            ref = city_cor(citya)
            bs = bootstrap(city_cor, citya, BasicSampling(n))
            test_bootsample(bs, ref, citya, n)
            test_confint(bs)
        end

        @testset "mean_and_sd: Vector input, 2 output variables" begin
            r = randn(25)
            ref = mean_and_std(r)
            bs = bootstrap(mean_and_std, r, BasicSampling(n))
            test_bootsample(bs, ref, r, n)
            test_confint(bs)
        end

        @testset "mean_and_sd: Student CI" begin
            r = randn(25)
            bs = bootstrap(mean_and_std, r, BasicSampling(n))
            ## Student confint
            cim = StudentConfInt()
            c = confint(bs, straps(bs, 2), cim, 1)
            @test c[1] >= c[2]  && c[1] <= c[3]
            @test c[1] ≈ original(bs, 1)
            @test level(cim) == 0.95
        end
    end

    @testset "Antithetic resampling" begin

        @testset "mean_and_sd: Vector input, 2 output variables" begin
            r = randn(50)
            ref = mean_and_std(r)
            bs = bootstrap(mean_and_std, r, AntitheticSampling(n))
            test_bootsample(bs, ref, r, n)
            test_confint(bs)
        end

    end

    @testset "Balanced resampling" begin

        @testset "city_ratio with DataFrame input" begin
            ref = city_ratio(city)
            @test ref ≈ 1.5203125
            bs = bootstrap(city_ratio, city, BalancedSampling(n))
            test_bootsample(bs, ref, city, n)
            test_confint(bs)
        end

        @testset "city_cor with DataFrame input" begin
            ref = city_cor(city)
            bs = bootstrap(city_cor, city, BalancedSampling(n))
            test_bootsample(bs, ref, city, n)
            test_confint(bs)
        end

        @testset "city_cor with DataArray input" begin
            ref = city_cor(citya)
            bs = bootstrap(city_cor, citya, BalancedSampling(n))
            test_bootsample(bs, ref, citya, n)
            test_confint(bs)
        end

        @testset "mean_and_sd: Vector input, 2 output variables" begin
            r = randn(50)
            ref = mean_and_std(r)
            bs = bootstrap(mean_and_std, r, BalancedSampling(n))
            test_bootsample(bs, ref, r, n)
            test_confint(bs)
            ## mean should be unbiased
            @test isapprox(bias(bs)[1], 0.0, atol = 1e-8)
        end

    end

    @testset "Exact resampling" begin

        nc = Bootstrap.nrun_exact(nrow(city))

        @testset "city_ratio with DataFrame input" begin
            ref = city_ratio(city)
            @test ref ≈ 1.5203125
            bs = bootstrap(city_ratio, city, ExactSampling())
            test_bootsample(bs, ref, city, nc)
            test_confint(bs)
        end

        @testset "mean: Vector input, 1 output variables" begin
            r = randn(10)
            ref = mean(r)
            bs = bootstrap(mean, r, ExactSampling())
            test_bootsample(bs, ref, r, nc)
            test_confint(bs)
        end

    end

    @testset "Maximum Entropy resampling" begin

        # Simulated AR(1) data copied from
        # https://github.com/colintbowers/DependentBootstrap.jl/blob/master/test/runtests.jl#L8

        nobs = 100

        function test_obs(n, seed = 1234)
            Random.seed!(seed)
            e = randn(n)
            x = zeros(Float64, n)
            for i = 2:n
                x[i] = 0.8 * x[i - 1] + e[i]
            end
            return sin.(x)
        end

        r = test_obs(nobs)
        ref = mean(r)
        s = MaximumEntropySampling(n)
        bs = bootstrap(mean, r, s)
        test_bootsample(bs, ref, r, n)
        test_confint(bs)

        # Collect the samples
        samples = zeros(eltype(r), (nobs, n))
        for i in 1:n
            draw!(s.cache, r, view(samples, :, i))
        end

        # Add some checks to ensure that our within sample variation is greater than our
        # across sample variation at any given "timestep".
        @test all(std(samples, dims = 2) .< std(r))
        @test mean(std(samples, dims = 2)) < 0.1  # NOTE: This is about 0.09 in julia and 0.08 in the R package
        @test std(r) > 0.5

    end

    @testset "ConfInt" begin
        @testset "All equal" begin
            r = rand(10)
            # In this case our count will always be the same because all values are positive
            bs = bootstrap(x->count(x .> 0.0), r, BasicSampling(n))
            test_confint_width0(bs)
        end
    end
end

end
