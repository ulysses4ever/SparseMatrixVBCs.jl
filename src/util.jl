using Base: @pure, @propagate_inbounds

@inline function fllog2(x::T) where {T <: Integer}
    return (sizeof(T) * 8 - 1) - leading_zeros(x)
end
@inline function cllog2(x::T) where {T <: Integer}
    return fllog2(x - 1) + 1
end

@inline function fld2(x::T) where {T <: Integer}
    return x >>> true
end
@inline function cld2(x::T) where {T <: Integer}
    return (x + true) >>> true
end

@pure nbits(::Type{T}) where {T} = sizeof(T) * 8
@pure log2nbits(::Type{T}) where {T} = flrlog2(nbits(T))

@inline function flpow1m(x::T) where {T <: Integer}
    return (1 << x) - 1
end

@inline undefs(T::Type, dims::Vararg{Any, N}) where {N} = Array{T, N}(undef, dims...)

zero!(arr) = fill!(arr, zero(eltype(arr)))