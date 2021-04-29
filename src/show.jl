const _show_pad = 12
const _show_signif = 5

function Base.show(io::IO, bs::BootstrapSample)
    r = _show_pad
    s = _show_signif
    df = estimate_summary(bs)
    df = split(string(df), '\n')[2:end]
    df[1] = replace(df[1], "Row" => "Var")
    println(io, "Bootstrap Sampling")
    println(io, rpad("  Estimates:", r))
    for i in 1:length(df)
        println(io, string("    ", df[i]))
    end
    println(io, rpad("  Sampling:", r),
            replace(string(typeof(sampling(bs))), "Bootstrap." => ""))
    println(io, rpad("  Samples:", r), nrun(bs))
    println(io, rpad("  Data:", r), data_summary(bs.data))
end


"""
Summarizes an object in the form of 'TYPE: { SIZE }'
"""
function data_summary(x)
    t = string(typeof(x))
    n = length(size(x)) > 0 ? size(x) : length(x)
    ns = join(n, " \u00D7 ") # Unicode 'times'
    s = "$t: { $ns }"
    return s
end

# TEMP hack to get `Base.show` to work on a `Tuple`
data_summary(x::Tuple) = ""


function estimate_summary(bs::BootstrapSample)
    n = nvar(bs)
    df = DataFrame(Estimate = [original(bs)...],
                   Bias = bias(bs),
                   StdError = stderror(bs))
    return df
end
