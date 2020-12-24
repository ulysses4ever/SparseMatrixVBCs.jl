using LinearAlgebra: Adjoint, Transpose
using Base: StridedVector, StridedMatrix, StridedVecOrMat
using SparseArrays: AdjOrTransStridedOrTriangularMatrix 
AdjOrTransSparseMatrix1DVBC{Ws, Tv, Ti} = Union{SparseMatrix1DVBC{Ws, Tv, Ti}, Adjoint{<:Any,<:SparseMatrix1DVBC{Ws, Tv, Ti}}, Transpose{<:Any, <:SparseMatrix1DVBC{Ws, Tv, Ti}}}

@generated function LinearAlgebra.mul!(y::StridedVector, A::SparseMatrix1DVBC{Ws, Tv, Ti}, x::StridedVector, α::Number, β::Number) where {Ws, Tv<:SIMD.VecTypes, Ti}
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
        if false && safe #TODO how do I zero out the other entries efficiently?
            thk = quote
                tmp = convert(Vec{$W, eltype(y)}, vload(Vec{$W, eltype(x)}, x, j))
            end
        else
            thk = quote
                tmp = convert(Vec{$W, eltype(y)}, Vec{$W, eltype(x)}(($(map(Δj -> :($Δj < w ? x[j + $Δj] : zero(eltype(x))), 0:W-1)...),)))
            end
        end
        thk = quote
            $thk
            q = A_ofs[l]
            for Q = A_pos[l]:(A_pos[l + 1] - 1)
                y[A_idx[Q]] += sum(convert(Vec{$W, eltype(y)}, vload(Vec{$W, Tv}, A_val, q)) * tmp)
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
            for l = 1:(L - $(max(Ws...)) - 1)
                j = Φ_spl[l]
                w = Φ_spl[l + 1] - j
                $(stripe_nest(true, Ws...))
            end
            for l = max(1, (L - $(max(Ws...)))):L
                j = Φ_spl[l]
                w = Φ_spl[l + 1] - j
                $(stripe_nest(false, Ws...))
            end
            return y
        end
    end

    return thunk
end

@generated function LinearAlgebra.mul!(y::StridedVector, adjA::Union{Adjoint{<:Any,<:SparseMatrix1DVBC{Ws, Tv, Ti}}, Transpose{<:Any, <:SparseMatrix1DVBC{Ws, Tv, Ti}}}, x::StridedVector, α::Number, β::Number) where {Ws, Tv<:SIMD.VecTypes, Ti}
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
                tmp += convert(Vec{$W, eltype(y)}, vload(Vec{$W, Tv}, A_val, q)) * convert(eltype(y), x[A_idx[Q]])
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
            for l = 1:(L - $(max(Ws...)) - 1)
                j = Φ_spl[l]
                w = Φ_spl[l + 1] - j
                $(stripe_nest(true, Ws...))
            end
            for l = max(1, (L - $(max(Ws...)))):L
                j = Φ_spl[l]
                w = Φ_spl[l + 1] - j
                $(stripe_nest(false, Ws...))
            end
            return y
        end
    end
    return thunk
end

Base.:*(adjA::AdjOrTransSparseMatrix1DVBC, x::StridedVector{Tx}) where {Tx} =
    (T = Base.promote_op(LinearAlgebra.matprod, eltype(adjA), Tx); mul!(similar(x, T, size(adjA, 1)), adjA, x, true, false))
Base.:*(adjA::AdjOrTransSparseMatrix1DVBC, B::AdjOrTransStridedOrTriangularMatrix) =
    (T = Base.promote_op(LinearAlgebra.matprod, eltype(adjA), eltype(B)); mul!(similar(B, T, (size(adjA, 1), size(B, 2))), adjA, B, true, false))