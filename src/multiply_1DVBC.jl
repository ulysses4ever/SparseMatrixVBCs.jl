using LinearAlgebra: Adjoint, Transpose
using Base: StridedVector, StridedMatrix, StridedVecOrMat
using SparseArrays: AdjOrTransStridedOrTriangularMatrix 
AdjOrTransSparseMatrix1DVBC{W, Tv, Ti} = Union{SparseMatrix1DVBC{W, Tv, Ti}, Adjoint{<:Any,<:SparseMatrix1DVBC{W, Tv, Ti}}, Transpose{<:Any, <:SparseMatrix1DVBC{W, Tv, Ti}}}

function LinearAlgebra.mul!(y::StridedVector, A::SparseMatrix1DVBC{W, Tv, Ti}, x::StridedVector, α::Number, β::Number) where {W, Tv<:SIMD.VecTypes, Ti}
    Δw = fld(CpuId.simdbytes(), sizeof(Tv))
    return _mul!(y, A, x, α, β, Val(Δw))
end
@generated function _mul!(y::StridedVector, A::SparseMatrix1DVBC{W, Tv, Ti}, x::StridedVector, α::Number, β::Number, ::Val{Δw}) where {W, Tv<:SIMD.VecTypes, Ti, Δw}
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
                y[A_idx[Q]] += sum(convert(Vec{$w, eltype(y)}, vload(Vec{$w, Tv}, A_val, q)) * tmp)
                q += w
            end
        end

        return thk
    end

    thunk = quote
        @inbounds begin
            size(A, 1) == size(y, 1) || throw(DimensionMismatch())
            size(A, 2) == size(x, 1) || throw(DimensionMismatch())
            (m, n) = size(A)

            yα = convert(eltype(y), α)

            if β != 1
                β != 0 ? rmul!(y, β) : fill!(y, zero(eltype(y)))
            end
            
            Φ_spl = A.Φ.spl
            A_pos = A.pos
            A_idx = A.idx
            A_ofs = A.ofs
            A_val = A.val
            L = length(A.Φ)
            L_safe = L
            while L_safe >= 1 && n + 1 - Φ_spl[L_safe] < Δw L_safe -= 1 end
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

function LinearAlgebra.mul!(y::StridedVector, adjA::Union{Adjoint{<:Any,<:SparseMatrix1DVBC{W, Tv, Ti}}, Transpose{<:Any, <:SparseMatrix1DVBC{W, Tv, Ti}}}, x::StridedVector, α::Number, β::Number) where {W, Tv<:SIMD.VecTypes, Ti}
    Δw = fld(CpuId.simdbytes(), sizeof(Tv))
    _mul!(y, adjA, x, α, β, Val(Δw))
end
@generated function _mul!(y::StridedVector, adjA::Union{Adjoint{<:Any,<:SparseMatrix1DVBC{W, Tv, Ti}}, Transpose{<:Any, <:SparseMatrix1DVBC{W, Tv, Ti}}}, x::StridedVector, α::Number, β::Number, ::Val{Δw}) where {W, Tv<:SIMD.VecTypes, Ti, Δw}
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
                tmp += convert(Vec{$w, eltype(y)}, vload(Vec{$w, Tv}, A_val, q)) * convert(eltype(y), x[A_idx[Q]])
                q += w
            end
        end
        if safe
            thk = quote
                $thk
                vstore(tmp, y, j)
            end
        else
            thk = quote
                $thk
                for Δj = 1:w
                    y[j + Δj - 1] = tmp[Δj]
                end
            end
        end

        return thk
    end

    thunk = quote
        @inbounds begin
            A = adjA.parent
            size(A, 2) == size(y, 1) || throw(DimensionMismatch())
            size(A, 1) == size(x, 1) || throw(DimensionMismatch())
            (m, n) = size(A)

            yα = convert(eltype(y), α)

            if β != 1
                β != 0 ? rmul!(y, β) : fill!(y, zero(eltype(y)))
            end
            
            Φ_spl = A.Φ.spl
            A_pos = A.pos
            A_idx = A.idx
            A_ofs = A.ofs
            A_val = A.val
            L = length(A.Φ)
            L_safe = L
            while L_safe >= 1 && n + 1 - Φ_spl[L_safe] < Δw L_safe -= 1 end
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

Base.:*(adjA::AdjOrTransSparseMatrix1DVBC, x::StridedVector{Tx}) where {Tx} =
    (T = Base.promote_op(LinearAlgebra.matprod, eltype(adjA), Tx); mul!(similar(x, T, size(adjA, 1)), adjA, x, true, false))
Base.:*(adjA::AdjOrTransSparseMatrix1DVBC, B::AdjOrTransStridedOrTriangularMatrix) =
    (T = Base.promote_op(LinearAlgebra.matprod, eltype(adjA), eltype(B)); mul!(similar(B, T, (size(adjA, 1), size(B, 2))), adjA, B, true, false))