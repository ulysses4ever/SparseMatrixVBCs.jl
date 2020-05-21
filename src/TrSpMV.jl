function TrSpMV!(y::Vector, A::SparseMatrixCSC, x::Vector)
    @inbounds begin
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
    function unsafe_thunk(W, tail...)
        return quote
            if w <= $W
                $(unsafe_thunk(W))
            else
                $(unsafe_thunk(tail...))
            end
        end
    end

    function unsafe_thunk(W)
        if W == 1
            return quote
                tmp = zero(eltype(y))
                ll = A_ofs[jj]
                for l = A_pos[jj]:(A_pos[jj + 1] - 1)
                    tmp += A_val[ll] * x[A_idx[l]]
                    ll += 1
                end
                y[i] = tmp
                nothing
            end
        else
            return quote
                tmp = Vec{$W, eltype(y)}(zero(eltype(y)))
                ll = A_ofs[jj]
                for l = A_pos[jj]:(A_pos[jj + 1] - 1)
                    tmp += vload(Vec{$W, eltype(y)}, A_val, ll) * x[A_idx[l]]
                    ll += w
                end
                vstore(tmp, y, i)
                nothing
            end
        end
    end

    function safe_thunk(W, tail...)
        return quote
            if w <= $W
                $(safe_thunk(W))
            else
                $(safe_thunk(tail...))
            end
        end
    end

    function safe_thunk(W)
        if W == 1
            return quote
                tmp = zero(eltype(y))
                ll = A_ofs[jj]
                for l = A_pos[jj]:(A_pos[jj + 1] - 1)
                    tmp += A_val[ll] * x[A_idx[l]]
                    ll += 1
                end
                y[i] = tmp
                nothing
            end
        else
            return quote
                tmp = Vec{$W, eltype(y)}(zero(eltype(y)))
                ll = A_ofs[jj]
                for l = A_pos[jj]:(A_pos[jj + 1] - 1)
                    tmp += vload(Vec{$W, eltype(y)}, A_val, ll) * x[A_idx[l]]
                    ll += w
                end
                for Δi = 1:w
                    y[i + Δi - 1] = tmp[Δi]
                end
                nothing
            end
        end
    end

    thunk = quote
        @inbounds begin
            size(A, 2) == size(y, 1) || throw(DimensionMismatch())
            size(A, 1) == size(x, 1) || throw(DimensionMismatch())
            m = length(y)
            n = length(x)
            
            A_spl = A.Π
            A_pos = A.pos
            A_idx = A.idx
            A_ofs = A.ofs
            A_val = A.val
            K = length(A_spl) - 1
            for jj = 1:(K - $(max(Ws...)) - 1)
                i = A_spl[jj]
                w = A_spl[jj + 1] - i
                $(unsafe_thunk(Ws...))
            end
            for jj = max(1, (K - $(max(Ws...)))):K
                i = A_spl[jj]
                w = A_spl[jj + 1] - i
                $(safe_thunk(Ws...))
            end
            return y
        end
    end
    return thunk
end