using Memoize, LinearAlgebra, CUDA

# All of these should work on both CPU and GPU

@memoize function powers(m; cuda=false)
    p = [i + j for i in 0:m, j in 0:m]
    if cuda
        cu(p)
    else
        p
    end
end

lhs(xs::CuArray, m) = dropdims(sum(reshape(xs, 1, 1, length(xs)) .^ powers(m; cuda=true), dims=3), dims=3)
lhs(xs, m) = dropdims(sum(reshape(xs, 1, 1, length(xs)) .^ powers(m), dims=3), dims=3)
rhs(xs, ys, m) = sum(xs .^ (0:m)' .* ys, dims=1)
polysolve(xs, ys, m) = lhs(xs, m) \ rhs(xs, ys, m)'

struct Polynomial{T<:AbstractArray}
    coefs::T
    powers::T
    function Polynomial(coefs::T) where {T<:AbstractArray}
        powers = similar(coefs)
        powers .= 0:(length(coefs)-1)
        new{T}(coefs, powers)
    end
end

(p::Polynomial)(x::AbstractVector) = sum((x' .^ p.powers) .* p.coefs, dims=1)