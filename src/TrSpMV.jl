function TrSpMV!(y::Vector, A::SparseMatrixCSC, x::Vector)
    @fastmath @inbounds begin
        size(A, 2) == size(y, 1) || throw(DimensionMismatch())
        size(A, 1) == size(x, 1) || throw(DimensionMismatch())
        m = length(y)
        n = length(x)
        
        A_pos = A.colptr
        A_idx = A.rowval
        A_val = A.nzval
        for i = 1:length(y)
            tmp = zero(eltype(y))
            for j = A_pos[i]:(A_pos[i + 1] - 1)
                tmp += A_val[j] * x[A_idx[j]]
            end
            y[i] = tmp
        end
        y
    end
end

@generated function TrSpMV!(y::Vector, A::SparseMatrix1DVBC{Ws, Tv, Ti}, x::Vector) where {Ws, Tv, Ti}
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
                tmp += vload(Vec{$W, eltype(y)}, A_val, q) * x[A_idx[Q]]
                q += w
            end
        end
        if safe
            thk = quote
                $thk
                vstore(tmp, y, i)
            end
        else
            thk = quote
                $thk
                for Δi = 1:w
                    y[i + Δi - 1] = tmp[Δi]
                end
            end
        end

        return thk
    end

    thunk = quote
        @fastmath @inbounds begin
            size(A, 2) == size(y, 1) || throw(DimensionMismatch())
            size(A, 1) == size(x, 1) || throw(DimensionMismatch())
            m = length(y)
            n = length(x)
            
            Φ_spl = A.Φ.spl
            A_pos = A.pos
            A_idx = A.idx
            A_ofs = A.ofs
            A_val = A.val
            L = length(A.Φ)
            for l = 1:(L - $(max(Ws...)) - 1)
                i = Φ_spl[l]
                w = Φ_spl[l + 1] - i
                $(stripe_nest(true, Ws...))
            end
            for l = max(1, (L - $(max(Ws...)))):L
                i = Φ_spl[l]
                w = Φ_spl[l + 1] - i
                $(stripe_nest(false, Ws...))
            end
            return y
        end
    end
    return thunk
end

using MacroTools

@generated function TrSpMV!(y::Vector, A::SparseMatrixVBC{U_max, Ws, Tv, Ti}, x::Vector) where {U_max, Ws, Tv, Ti}
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
        thk = :(tmp += vload(Vec{$W, eltype(y)}, A_val, q) * x[j])
        for Δj = 1:U-1
            thk = quote
                $thk
                tmp += vload(Vec{$W, eltype(y)}, A_val, q + w * $Δj) * x[j + $Δj]
            end
        end
        return thk
    end

    thunk = quote
        @fastmath @inbounds begin
            size(A, 2) == size(y, 1) || throw(DimensionMismatch())
            size(A, 1) == size(x, 1) || throw(DimensionMismatch())
            m = length(y)
            n = length(x)
            
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