module RFIKiller

include("polynomials.jl")
include("filters.jl")
include("masked_stats.jl")
include("dada_filter.jl")

# Precompile to get around latency
precompile(kill_rfi!, (Matrix{Float32}, Matrix{Bool}))
precompile(kill_rfi!, (CuMatrix{Float32}, CuMatrix{Bool}))

export julia_main

end
