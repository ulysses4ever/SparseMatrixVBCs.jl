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

le_nest(f, x, ns) = _le_nest(f, x, ns...)
_le_nest(f, x, n) = f(n)
function _le_nest(f, x, n, ns...)
    return quote
        if $x <= $n
            $(f(n))
        else
            $(_le_nest(f, x, ns...))
        end
    end
end

eq_nest(f, x, ns) = _eq_nest(f, x, ns...)
_eq_nest(f, x, n) = f(n)
function _eq_nest(f, x, n, ns...)
    return quote
        if $x == $n
            $(f(n))
        else
            $(_eq_nest(f, x, ns...))
        end
    end
end