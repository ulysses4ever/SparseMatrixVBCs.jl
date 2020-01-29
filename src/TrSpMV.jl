function TrSpMV!(y::Vector, A::SparseMatrixCSC, x::Vector)
    @inbounds begin
        size(A, 2) == size(y, 1) || throw(DimensionMismatch())
        size(A, 1) == size(x, 1) || throw(DimensionMismatch())
        m = length(y)
        n = length(x)
        
        pos = A.colptr
        idx = A.rowval
        val = A.nzval
        for i = 1:length(y)
            tmp = zero(eltype(y))
            for j = pos[i]:(pos[i + 1] - 1)
                tmp += val[j] * x[idx[j]]
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
                qq = ofs[jj]
                for q = pos[jj]:(pos[jj + 1] - 1)
                    tmp += val[qq] * x[idx[q]]
                    qq += 1
                end
                y[i] = tmp
                nothing
            end
        else
            return quote
                tmp = Vec{$W, eltype(y)}(zero(eltype(y)))
                qq = ofs[jj]
                for q = pos[jj]:(pos[jj + 1] - 1)
                    tmp += vload(Vec{$W, eltype(y)}, val, qq) * x[idx[q]]
                    qq += w
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
                qq = ofs[jj]
                for q = pos[jj]:(pos[jj + 1] - 1)
                    tmp += val[qq] * x[idx[q]]
                    qq += 1
                end
                y[i] = tmp
                nothing
            end
        else
            return quote
                tmp = Vec{$W, eltype(y)}(zero(eltype(y)))
                qq = ofs[jj]
                for q = pos[jj]:(pos[jj + 1] - 1)
                    tmp += vload(Vec{$W, eltype(y)}, val, qq) * x[idx[q]]
                    qq += w
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
            
            spl = A.spl
            pos = A.pos
            idx = A.idx
            ofs = A.ofs
            val = A.val
            k = length(spl) - 1
            for jj = 1:(k - $(max(Ws...)) - 1)
                i = spl[jj]
                w = spl[jj + 1] - i
                $(unsafe_thunk(Ws...))
            end
            for jj = max(1, (k - $(max(Ws...)))):k
                i = spl[jj]
                w = spl[jj + 1] - i
                $(safe_thunk(Ws...))
            end
            return y
        end
    end
    return thunk
end