module RFIKiller

using LoopVectorization, StatsBase
import Polynomials

export kill_rfi!

# Data is (channels,samples) as consecutive frequencies as consecutive in memorys

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

    @turbo @. acc / len
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

    @turbo @. σ² / (len - 1)
end

function bandpass!(spectra, mask, tolerance=0.001)
    bp = masked_mean(spectra, mask, 2)
    not_nans = findall(x -> !isnan(x), bp)
    ms = bp .< (tolerance .* median(@view bp[not_nans]))
    @views spectra[not_nans, :] ./= bp[not_nans]
    @. mask[ms, :] = false
    nothing
end

function sigmacut!(spectra, mask, ax, σ=3)
    σs = sqrt.(masked_var(spectra, mask, ax))
    not_nans = findall(x -> !isnan(x), σs)
    σs_good = @view σs[not_nans]
    μ_σs = mean(σs_good)
    σ_σs = std(σs_good, mean=μ_σs)
    ms = @. σs > μ_σs + σ * σ_σs
    if ax == 1
        @. mask[:, ms] = false
    else
        @. mask[ms, :] = false
    end
    nothing
end

function detrend!(spectra, mask, axis, degree)
    @assert axis == 1 || axis == 2 "Axis must be either 1 or 2"

    ax_mean = masked_mean(spectra, mask, axis)
    not_nans = findall(x -> !isnan(x), ax_mean)
    ax_mean_good = @view ax_mean[not_nans]

    if axis == 1
        xs = axes(spectra, 2)
    else
        xs = axes(spectra, 1)
    end

    f = Polynomials.fit(xs[not_nans], ax_mean_good, degree)

    if axis == 1
        @. spectra -= f(xs)'
    else
        @. spectra -= f(xs)
    end

    nothing
end

function zero_dm_filter!(spectra, mask, σ_limit)
    dmzero = masked_mean(spectra, mask, 2)
    not_nans = findall(x -> !isnan(x), dmzero)
    dmzero = dmzero .- median(dmzero[not_nans])
    σ = 1.4826 * mad(dmzero[not_nans])

    # Find outliers
    ms = @. abs(dmzero) > σ_limit * σ
    @. mask[ms, :] = false
end

function kill_rfi!(spectra::AbstractMatrix)
    mask = ones(Bool, size(spectra))
    bandpass!(spectra, mask, 0.001)
    sigmacut!(spectra, mask, 1, 3)
    sigmacut!(spectra, mask, 2, 6)
    zero_dm_filter!(spectra, mask, 7)
    detrend!(spectra, mask, 1, 4)
    detrend!(spectra, mask, 2, 6)
    spectra[.!mask] .= mean(spectra[mask])
    mask
end

end
