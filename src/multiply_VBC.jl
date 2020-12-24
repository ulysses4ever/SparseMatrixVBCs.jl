AdjOrTransSparseMatrixVBC{U_max, Ws, Tv, Ti} = Union{SparseMatrixVBC{U_max, Ws, Tv, Ti}, Adjoint{<:Any,<:SparseMatrixVBC{U_max, Ws, Tv, Ti}}, Transpose{<:Any, <:SparseMatrixVBC{U_max, Ws, Tv, Ti}}}

@generated function LinearAlgebra.mul!(y::StridedVector, adjA::Union{Adjoint{<:Any,<:SparseMatrixVBC{U_max, Ws, Tv, Ti}}, Transpose{<:Any, <:SparseMatrixVBC{U_max, Ws, Tv, Ti}}}, x::StridedVector, α::Number, β::Number) where {U_max, Ws, Tv<:SIMD.VecTypes, Ti}
    stripe_nest(safe) = stripe_nest(safe, Ws...)
    stripe_nest(safe, W) = stripe_body(safe, W)
    function stripe_nest(safe, W, tail...)
        return quote
            if w <= $W
                $(stripe_body(safe, W))
            else
                $(stripe_nest(safe, tail...))
            end
        end
    end

    function stripe_body(safe, W)
        thk = quote
            tmp = Vec{$W, eltype(y)}(zero(eltype(y)))
            q = A_ofs[l]
            for Q = A_pos[l]:(A_pos[l + 1] - 1)
                k = A_idx[Q]
                j = Π_spl[k]
                u = Π_spl[k + 1] - j
                $(block_nest(W))
                q += u * w
            end
        end
        if safe
            return quote
                $thk
                vstore(tmp, y, i)
            end
        else
            return quote
                $thk
                for Δi = 0:w - 1
                    y[i + Δi] = tmp[1 + Δi]
                end
            end
        end
    end

    block_nest(W) = block_nest(W, U_max)
    function block_nest(W, U)
        if U == 1
            return block_body(W, U)
        else
            return quote
                if u == $U
                    $(block_body(W, U))
                else
                    $(block_nest(W, U - 1))
                end
            end
        end
    end

    function block_body(W, U)
        thk = :()
        for Δj = 0:U-1
            thk = quote
                $thk
                tmp += convert(Vec{$W, eltype(y)}, vload(Vec{$W, Tv}, A_val, q + w * $Δj)) * convert(eltype(y), x[j + $Δj])
            end
        end
        return thk
    end

    thunk = quote
        @inbounds begin
            A = adjA.parent
            size(A, 2) == size(y, 1) || throw(DimensionMismatch())
            size(A, 1) == size(x, 1) || throw(DimensionMismatch())
            m = length(y)
            n = length(x)

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
            while L_safe > 1 && n + 1 - Φ_spl[L_safe] < $(max(Ws...)) L_safe -= 1 end
            for l = 1:(L_safe - 1)
                i = Φ_spl[l]
                w = Φ_spl[l + 1] - i
                $(stripe_nest(true))
            end
            for l = L_safe:L
                i = Φ_spl[l]
                w = Φ_spl[l + 1] - i
                $(stripe_nest(false))
            end
            return y
        end
    end
    return thunk
end

Base.:*(adjA::AdjOrTransSparseMatrixVBC, x::StridedVector{Tx}) where {Tx} =
    (T = Base.promote_op(LinearAlgebra.matprod, eltype(adjA), Tx); mul!(similar(x, T, size(adjA, 1)), adjA, x, true, false))
Base.:*(adjA::AdjOrTransSparseMatrixVBC, B::AdjOrTransStridedOrTriangularMatrix) =
    (T = Base.promote_op(LinearAlgebra.matprod, eltype(adjA), eltype(B)); mul!(similar(B, T, (size(adjA, 1), size(B, 2))), adjA, B, true, false))