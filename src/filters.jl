# Data is (channels,samples) as consecutive frequencies as consecutive in memorys

function bandpass!(spectra, mask, tolerance=0.001)
    bp = masked_mean(spectra, mask, 2)
    not_nans = findall(x -> !isnan(x), bp)
    ms = bp .< (tolerance .* median(@view bp[not_nans]))
    @views spectra[not_nans, :] ./= bp[not_nans]
    mask[ms, :] .= false
    nothing
end

function sigmacut!(spectra, mask, ax, σ=3)
    σs = sqrt.(masked_var(spectra, mask, ax))
    not_nans = findall(x -> !isnan(x), σs)
    μ_σs = mean(σs[not_nans])
    σ_σs = std(σs[not_nans], mean=μ_σs)
    ms = @. σs > μ_σs + σ * σ_σs
    if ax == 1
        mask[:, ms] .= false
    else
        mask[ms, :] .= false
    end
    nothing
end

function detrend!(spectra, mask, axis, degree)
    @assert axis == 1 || axis == 2 "Axis must be either 1 or 2"

    ax_mean = masked_mean(spectra, mask, axis)
    not_nans = findall(x -> !isnan(x), ax_mean)

    if axis == 1
        xs = similar(spectra, eltype(spectra), size(spectra)[2])
        xs .= range(0, 1, size(spectra)[2])
    else
        xs = similar(spectra, eltype(spectra), size(spectra)[1])
        xs .= range(0, 1, size(spectra)[1])
    end

    f = Polynomial(polysolve(xs[not_nans], ax_mean[not_nans], degree))

    if axis == 1
        spectra .-= f(xs)
    else
        spectra .-= f(xs)'
    end

    nothing
end

function zero_dm_filter!(spectra, mask, σ_limit)
    dmzero = masked_mean(spectra, mask, 2)
    not_nans = findall(x -> !isnan(x), dmzero)
    dmzero_good = dmzero[not_nans]
    good_median = median(dmzero_good)
    dmzero = dmzero .- good_median
    σ = median(@. abs(dmzero_good - good_median))
    # Find outliers
    ms = @. abs(dmzero) > σ_limit * σ
    @. mask[ms, :] = false
    nothing
end

function kill_rfi!(spectra, mask)
    # Reset mask
    mask .= true
    # Apply filters
    bandpass!(spectra, mask, 0.001)
    sigmacut!(spectra, mask, 1, 3)
    sigmacut!(spectra, mask, 2, 6)
    zero_dm_filter!(spectra, mask, 7)
    detrend!(spectra, mask, 1, 4)
    detrend!(spectra, mask, 2, 6)
    # Mask to mean
    spectra[.!mask] .= mean(spectra[mask])
    nothing
end