AdjOrTransSparseMatrixVBC{U, W, Tv, Ti} = Union{SparseMatrixVBC{U, W, Tv, Ti}, Adjoint{<:Any,<:SparseMatrixVBC{U, W, Tv, Ti}}, Transpose{<:Any, <:SparseMatrixVBC{U, W, Tv, Ti}}}

function LinearAlgebra.mul!(y::StridedVector, A::SparseMatrixVBC{U, W, Tv, Ti}, x::StridedVector, α::Number, β::Number) where {U, W, Tv<:SIMD.VecTypes, Ti}
    Δw = fld(DEFAULT_SIMD_SIZE, sizeof(eltype(y)))
    return _mul!(y, A, x, α, β, Val(Δw))
end
@generated function _mul!(y::StridedVector, A::SparseMatrixVBC{U, W, Tv, Ti}, x::StridedVector, α::Number, β::Number, ::Val{Δw}) where {U, W, Δw, Tv<:SIMD.VecTypes, Ti}
    if Δw == 1
        ws = (1:W...,)
    else
        ws = (1, Δw : Δw : (W + Δw - 1)...,)
    end

    function stripe_body(safe, w)
        if false && safe #TODO how do I zero out the other entries efficiently?
            thk = quote
                tmp = convert(Vec{$w, eltype(y)}, vload(Vec{$w, eltype(x)}, x, j))
            end
        else
            thk = quote
                tmp = convert(Vec{$w, eltype(y)}, Vec{$w, eltype(x)}(($(map(Δj -> :($Δj < w ? x[j + $Δj] : zero(eltype(x))), 0:w-1)...),)))
            end
        end
        thk = quote
            $thk
            q = A_ofs[l]
            for Q = A_pos[l]:(A_pos[l + 1] - 1)
                k = A_idx[Q]
                i = Π_spl[k]
                u = Π_spl[k + 1] - i
                $(eq_nest(u->block_body(w, u), :u, 1:U))
                q += u * w
            end
        end
        return thk
    end

    function block_body(w, u)
        thk = :()
        for Δi = 0:u-1
            thk = quote
                $thk
                y[i + $Δi] += sum(convert(Vec{$w, eltype(y)}, vload(Vec{$w, Tv}, A_val, q + w * $Δi)) * tmp)
            end
        end
        return thk
    end

    thunk = quote
        @inbounds begin
            size(A, 1) == size(y, 1) || throw(DimensionMismatch())
            size(A, 2) == size(x, 1) || throw(DimensionMismatch())
            (m, n) = size(A)

            if β != 1
                β != 0 ? rmul!(y, β) : fill!(y, zero(eltype(y)))
            end
            
            Π_spl = A.Π.spl
            Φ_spl = A.Φ.spl
            A_pos = A.pos
            A_idx = A.idx
            A_ofs = A.ofs
            A_val = A.val
            L = length(A.Φ)
            L_safe = L
            while L_safe >= 1 && n + 1 - Φ_spl[L_safe] < $(Δw * cld(W, Δw)) L_safe -= 1 end
            for l = 1:L_safe
                j = Φ_spl[l]
                w = Φ_spl[l + 1] - j
                $(le_nest(w->stripe_body(true, w), :w, ws))
            end
            for l = L_safe + 1:L
                j = Φ_spl[l]
                w = Φ_spl[l + 1] - j
                $(le_nest(w->stripe_body(false, w), :w, ws))
            end
            return y
        end
    end
    if eltype(y) <: SIMD.FloatingTypes
        thunk = quote
            @fastmath $thunk
        end
    end
    return thunk
end

function LinearAlgebra.mul!(y::StridedVector, adjA::Union{Adjoint{<:Any,<:SparseMatrixVBC{U, W, Tv, Ti}}, Transpose{<:Any, <:SparseMatrixVBC{U, W, Tv, Ti}}}, x::StridedVector, α::Number, β::Number) where {U, W, Tv<:SIMD.VecTypes, Ti}
    Δw = fld(DEFAULT_SIMD_SIZE, sizeof(eltype(y)))
    return _mul!(y, adjA, x, α, β, Val(Δw))
end
@generated function _VBR_mul!(y::StridedVector, x::StridedVector, α::Number, β::Number, l, j, w, A_ofs, A_pos, A_idx, A_val, Π_spl, ::Val{Δw}, ::Val{U}, ::Val{W}, ::Val{Tv}, ::Val{Ti}) where {Δw, U, W, Tv, Ti}
    if Δw == 1
        ws = (1:W...,)
    else
        ws = (1, Δw : Δw : (W + Δw - 1)...,)
    end
    function stripe_body(safe, w)
        thk = quote
            tmp = Vec{$w, eltype(y)}(zero(eltype(y)))
            q = A_ofs[l]
            for Q = A_pos[l]:(A_pos[l + 1] - 1)
                k = A_idx[Q]
                i = Π_spl[k]
                u = Π_spl[k + 1] - i
                $(eq_nest(u->block_body(w, u), :u, 1:U))
                q += u * w
            end
        end
        if safe
            return quote
                $thk
                vstore(tmp, y, j)
            end
        else
            return quote
                $thk
                for Δj = 0:(w - 1)
                    y[j + Δj] = tmp[1 + Δj]
                end
            end
        end
    end

    function block_body(w, u)
        thk = :()
        for Δi = 0:u-1
            thk = quote
                $thk
                tmp += convert(Vec{$w, eltype(y)}, vload(Vec{$w, Tv}, A_val, q + w * $Δi)) * convert(eltype(y), x[i + $Δi])
            end
        end
        return thk
    end
    thunk = quote
        @inbounds begin
            $(le_nest(w->stripe_body(false, w), :w, ws))
        end
    end

    if eltype(y) <: SIMD.FloatingTypes
        thunk = :(@fastmath $thunk)
    end

    return :(Base.@_inline_meta; $thunk)
end

@inline function _mul!(y::StridedVector, adjA::Union{Adjoint{<:Any,<:SparseMatrixVBC{U, W, Tv, Ti}}, Transpose{<:Any, <:SparseMatrixVBC{U, W, Tv, Ti}}}, x::StridedVector, α::Number, β::Number, Δw) where {U, W, Tv<:SIMD.VecTypes, Ti}
    @inbounds begin
        A = adjA.parent
        size(A, 2) == size(y, 1) || throw(DimensionMismatch())
        size(A, 1) == size(x, 1) || throw(DimensionMismatch())
        (m, n) = size(A)

        if β != 1
            β != 0 ? rmul!(y, β) : fill!(y, zero(eltype(y)))
        end
        
        Π_spl = A.Π.spl
        Φ_spl = A.Φ.spl
        A_pos = A.pos
        A_idx = A.idx
        A_ofs = A.ofs
        A_val = A.val
        L = length(A.Φ)
        #=
        L_safe = L
        while L_safe >= 1 && n + 1 - Φ_spl[L_safe] < $(Δw * cld(W, Δw)) L_safe -= 1 end
        for l = 1:L_safe
            j = Φ_spl[l]
            w = Φ_spl[l + 1] - j
            $(le_nest(w->stripe_body(true, w), :w, ws))
        end
        for l = L_safe + 1:L
            j = Φ_spl[l]
            w = Φ_spl[l + 1] - j
            $(le_nest(w->stripe_body(false, w), :w, ws))
        end
        =#

        l′ = Atomic{Int}(1)
        @threads for t = 1:nthreads()
            while (l = atomic_add!(l′, 1)) <= L
                j = Φ_spl[l]
                w = Φ_spl[l + 1] - j
                _VBR_mul!(y, x, α, β, l, j, w, A_ofs, A_pos, A_idx, A_val, Π_spl, Δw, Val(U), Val(W), Val(Tv), Val(Ti))
            end
        end
        return y
    end
end

Base.:*(adjA::AdjOrTransSparseMatrixVBC, x::StridedVector{Tx}) where {Tx} =
    (T = Base.promote_op(LinearAlgebra.matprod, eltype(adjA), Tx); mul!(similar(x, T, size(adjA, 1)), adjA, x, true, false))
Base.:*(adjA::AdjOrTransSparseMatrixVBC, B::AdjOrTransDenseMatrix) =
    (T = Base.promote_op(LinearAlgebra.matprod, eltype(adjA), eltype(B)); mul!(similar(B, T, (size(adjA, 1), size(B, 2))), adjA, B, true, false))