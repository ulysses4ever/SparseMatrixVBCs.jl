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
                q = A_ofs[l]
                for Q = A_pos[l]:(A_pos[l + 1] - 1)
                    tmp += A_val[q] * x[A_idx[Q]]
                    q += 1
                end
                y[i] = tmp
                nothing
            end
        else
            return quote
                tmp = Vec{$W, eltype(y)}(zero(eltype(y)))
                q = A_ofs[l]
                for Q = A_pos[l]:(A_pos[l + 1] - 1)
                    tmp += vload(Vec{$W, eltype(y)}, A_val, q) * x[A_idx[Q]]
                    q += w
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
                q = A_ofs[l]
                for Q = A_pos[l]:(A_pos[l + 1] - 1)
                    tmp += A_val[q] * x[A_idx[Q]]
                    q += 1
                end
                y[i] = tmp
                nothing
            end
        else
            return quote
                tmp = Vec{$W, eltype(y)}(zero(eltype(y)))
                q = A_ofs[l]
                for Q = A_pos[l]:(A_pos[l + 1] - 1)
                    tmp += vload(Vec{$W, eltype(y)}, A_val, q) * x[A_idx[Q]]
                    q += w
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
            
            A_spl = A.spl
            A_pos = A.pos
            A_idx = A.idx
            A_ofs = A.ofs
            A_val = A.val
            L = length(A_spl) - 1
            for l = 1:(L - $(max(Ws...)) - 1)
                i = A_spl[l]
                w = A_spl[l + 1] - i
                $(unsafe_thunk(Ws...))
            end
            for l = max(1, (L - $(max(Ws...)))):L
                i = A_spl[l]
                w = A_spl[l + 1] - i
                $(safe_thunk(Ws...))
            end
            return y
        end
    end
    return thunk
end

@generated function TrSpMV!(y::Vector, A::SparseMatrixVBC{Us, Ws, Tv, Ti}, x::Vector) where {Us, Ws, Tv, Ti}
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
                q = A_ofs[l]
                for Q = A_pos[l]:(A_pos[l + 1] - 1)
                    tmp += A_val[q] * x[A_idx[Q]]
                    q += 1
                end
                y[i] = tmp
                nothing
            end
        else
            return quote
                tmp = Vec{$W, eltype(y)}(zero(eltype(y)))
                q = A_ofs[l]
                for Q = A_pos[l]:(A_pos[l + 1] - 1)
                    tmp += vload(Vec{$W, eltype(y)}, A_val, q) * x[A_idx[Q]]
                    q += w
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
                q = A_ofs[l]
                for Q = A_pos[l]:(A_pos[l + 1] - 1)
                    tmp += A_val[q] * x[A_idx[Q]]
                    q += 1
                end
                y[i] = tmp
                nothing
            end
        else
            return quote
                tmp = Vec{$W, eltype(y)}(zero(eltype(y)))
                q = A_ofs[l]
                for Q = A_pos[l]:(A_pos[l + 1] - 1)
                    tmp += vload(Vec{$W, eltype(y)}, A_val, q) * x[A_idx[Q]]
                    q += w
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
            
            A_spl = A.spl
            A_pos = A.pos
            A_idx = A.idx
            A_ofs = A.ofs
            A_val = A.val
            L = length(A_spl) - 1
            for l = 1:(L - $(max(Ws...)) - 1)
                i = A_spl[l]
                w = A_spl[l + 1] - i
                $(unsafe_thunk(Ws...))
            end
            for l = max(1, (L - $(max(Ws...)))):L
                i = A_spl[l]
                w = A_spl[l + 1] - i
                $(safe_thunk(Ws...))
            end
            return y
        end
    end
    return thunk
end