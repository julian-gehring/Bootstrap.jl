module TestBootsampleNonParametric

using Bootstrap
using Bootstrap.Datasets
using Base.Test

using DataFrames
using StatsBase


@testset "Non-parametric bootstraps" begin

    function test_bootsample(bs, ref, raw_data, n)

        show(IOBuffer(), bs)
        @test issubtype(typeof(bs), NonParametricBootstrapSample)
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
        @test length(se(bs)) == length(ref)
        [@test s >= 0 for s in se(bs)]

        [@test original(bs, i) == original(bs)[i]  for i in 1:nvar(bs)]
        [@test straps(bs, i) == straps(bs)[i]  for i in 1:nvar(bs)]
        [@test bias(bs, i) == bias(bs)[i]  for i in 1:nvar(bs)]
        [@test se(bs, i) == se(bs)[i]  for i in 1:nvar(bs)]

        @test_throws MethodError model(bs)

        return Void
    end

    function test_ci(bs)

        cim_all = (BasicConfInt(), PercentileConfInt(), NormalConfInt(), BCaConfInt())
        for cim in cim_all
            c = ci(bs, cim)
            [@test (x[1] >= x[2]  && x[1] <= x[3]) for x in c]
            [@test x[1] ≈ t0 for (x, t0) in zip(c, original(bs))]
            @test level(cim) == 0.95
        end

        return Void
    end

    n = 250

    ## 'city' dataset
    citya = convert(Array, city)

    city_ratio(df::DataFrames.DataFrame) = mean(df[:,:X]) ./ mean(df[:,:U])
    city_ratio(a::AbstractArray) = mean(a[:,2]) ./ mean(a[:,1])

    city_cor(x::AbstractArray) = cor(x[:,1], x[:,2])
    city_cor(x::AbstractDataFrame) = cor(x[:,:X], x[:,:U])


    @testset "Basic resampling" begin

        @testset "city_ratio with DataFrame input" begin
            ref = city_ratio(city)
            @test ref  ≈ 1.5203125
            bs = bootstrap(city, city_ratio, BasicSampling(n))
            test_bootsample(bs, ref, city, n)
            test_ci(bs)
        end

        @testset "city_cor with DataFrame input" begin
            ref = city_cor(city)
            bs = bootstrap(city, city_cor, BasicSampling(n))
            test_bootsample(bs, ref, city, n)
            test_ci(bs)
        end

        @testset "city_cor with DataArray input" begin
            ref = city_cor(citya)
            bs = bootstrap(citya, city_cor, BasicSampling(n))
            test_bootsample(bs, ref, citya, n)
            test_ci(bs)
        end

        @testset "mean_and_sd: Vector input, 2 output variables" begin
            r = randn(25)
            ref = mean_and_std(r)
            bs = bootstrap(r, mean_and_std, BasicSampling(n))
            test_bootsample(bs, ref, r, n)
            test_ci(bs)
        end

        @testset "mean_and_sd: Student CI" begin
            r = randn(25)
            bs = bootstrap(r, mean_and_std, BasicSampling(n))
            ## Student confint
            cim = StudentConfInt()
            c = ci(bs, straps(bs, 2), cim, 1)
            @test c[1] >= c[2]  && c[1] <= c[3]
            @test c[1] ≈ original(bs, 1)
            @test level(cim) == 0.95
        end

    end

    @testset "Antithetic resampling" begin

        @testset "mean_and_sd: Vector input, 2 output variables" begin
            r = randn(50)
            ref = mean_and_std(r)
            bs = bootstrap(r, mean_and_std, AntitheticSampling(n))
            test_bootsample(bs, ref, r, n)
            test_ci(bs)
        end

    end

    @testset "Balanced resampling" begin

        @testset "city_ratio with DataFrame input" begin
            ref = city_ratio(city)
            @test ref ≈ 1.5203125
            bs = bootstrap(city, city_ratio, BalancedSampling(n))
            test_bootsample(bs, ref, city, n)
            test_ci(bs)
        end

        @testset "city_cor with DataFrame input" begin
            ref = city_cor(city)
            bs = bootstrap(city, city_cor, BalancedSampling(n))
            test_bootsample(bs, ref, city, n)
            test_ci(bs)
        end

        @testset "city_cor with DataArray input" begin
            ref = city_cor(citya)
            bs = bootstrap(citya, city_cor, BalancedSampling(n))
            test_bootsample(bs, ref, citya, n)
            test_ci(bs)
        end

        @testset "mean_and_sd: Vector input, 2 output variables" begin
            r = randn(50)
            ref = mean_and_std(r)
            bs = bootstrap(r, mean_and_std, BalancedSampling(n))
            test_bootsample(bs, ref, r, n)
            test_ci(bs)
            ## mean should be unbiased
            @test isapprox( bias(bs)[1], 0.0, atol = 1e-8 )
        end

    end

    @testset "Exact resampling" begin

        nc = Bootstrap.nrun_exact(nrow(city))

        @testset "city_ratio with DataFrame input" begin
            ref = city_ratio(city)
            @test ref ≈ 1.5203125
            bs = bootstrap(city, city_ratio, ExactSampling())
            test_bootsample(bs, ref, city, nc)
            test_ci(bs)
        end

        @testset "mean: Vector input, 1 output variables" begin
            r = randn(10)
            ref = mean(r)
            bs = bootstrap(r, mean, ExactSampling())
            test_bootsample(bs, ref, r, nc)
            test_ci(bs)
        end

    end

    @testset "Maximum Entropy resampling" begin

        # Simulated AR(1) data copied from
        # https://github.com/colintbowers/DependentBootstrap.jl/blob/master/test/runtests.jl#L8

        nobs = 100

        function test_obs(n, seed=1234)
            srand(seed)
            e = randn(n)
            x = Array{Float64}(n)
            x[1] = 0.0
            for i = 2:n
                x[i] = 0.8 * x[i-1] + e[i]
            end
            return sin.(x)
        end

        r = test_obs(nobs)
        ref = mean(r)
        s = MaximumEntropySampling(n)
        bs = bootstrap(r, mean, s)
        test_bootsample(bs, ref, r, n)
        test_ci(bs)

        # Collect the samples
        samples = zeros(eltype(r), (nobs, n))
        for i in 1:n
            # For some reason the samples are only filled in if we have
            # an explicit assignment back into the matrix.
            samples[:, i] = draw!(s.cache, r, samples[:, i])
        end

        # Add some checks to ensure that our within sample variation is greater than our
        # across sample variation at any given "timestep".
        @test all(std(samples, 2) .< std(r))
        @test mean(std(samples, 2)) < 0.1  # NOTE: This is about 0.09 in julia and 0.08 in the R package
        @test std(r) > 0.5

    end

end

end
