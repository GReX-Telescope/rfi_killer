using LoopVectorization, StatsBase

function masked_mean(x::AbstractMatrix{T}, mask, axis) where {T<:Number}
    @assert size(x) == size(mask) "The mask and input must have the same size"
    @assert axis == 1 || axis == 2 "Axis must be either 1 or 2"

    if axis == 1
        ax = axes(x, 2)
    else
        ax = axes(x, 1)
    end

    len = zeros(Int32, length(ax))
    acc = zeros(T, length(ax))

    if axis == 1
        @turbo for j in axes(x, 2), i in axes(x, 1)
            acc[j] += x[i, j] * mask[i, j]
            len[j] += mask[i, j]
        end
    else
        @turbo for j in axes(x, 2), i in axes(x, 1)
            acc[i] += x[i, j] * mask[i, j]
            len[i] += mask[i, j]
        end
    end

    # Use acc as our output vector instead of making a new one
    @turbo acc .= acc ./ len
end

function masked_var(x::AbstractMatrix{T}, mask, axis) where {T<:Number}
    @assert size(x) == size(mask) "The mask and input must have the same size"
    @assert axis == 1 || axis == 2 "Axis must be either 1 or 2"

    x̄ = masked_mean(x, mask, axis)

    if axis == 1
        ax = axes(x, 2)
    else
        ax = axes(x, 1)
    end

    len = zeros(Int32, length(ax))
    σ² = zeros(T, length(ax))

    if axis == 1
        @turbo for j in axes(x, 2), i in axes(x, 1)
            σ²[j] += abs2(x[i, j] - x̄[j]) * mask[i, j]
            len[j] += mask[i, j]
        end
    else
        @turbo for j in axes(x, 2), i in axes(x, 1)
            σ²[i] += abs2(x[i, j] - x̄[i]) * mask[i, j]
            len[i] += mask[i, j]
        end
    end

    @turbo σ² .= σ² ./ (len - 1)
end