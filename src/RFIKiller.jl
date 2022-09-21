module RFIKiller

include("filters.jl")
include("masked_stats.jl")
include("dada_filter.jl")

# Precompile to get around the LoopVectorization latency
precompile(kill_rfi!, (Matrix{Float32}, Matrix{Bool}))

export julia_main

end
