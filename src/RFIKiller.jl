module RFIKiller

include("filters.jl")
include("masked_stats.jl")
include("dada_filter.jl")

export kill_rfi!

end
