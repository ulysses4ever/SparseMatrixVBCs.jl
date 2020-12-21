function SparseMatrix1DVBC{Ws}(A::SparseMatrixCSC{Tv, Ti}, method=DynamicTotalChunker(model_SparseMatrix1DVBC_memory(Tv, Ti), max(Ws...))) where {Ws, Tv, Ti}
    x_net = Ref(Ti[])
    Φ = pack_stripe(A, method, x_net=x_net)
    if length(x_net) == 0
        return SparseMatrix1DVBC{Ws}(A, Φ)
    else
        L = length(Φ)
        Φ = convert(SplitPartition, Φ)
        spl = Φ.spl
        pos = undefs(Ti, L + 1)
        ofs = undefs(Ti, L + 1)
        pos[1] = 1
        ofs[1] = 1
        for l = 1:L
            pos[l + 1] = pos[l] + x_net[][l]
            ofs[l + 1] = ofs[l] + x_net[][l] * (spl[l + 1] - spl[l])
        end
        return _construct_SparseMatrix1DVBC(Val(Ws), A, Φ, pos, ofs)
    end
end

function SparseMatrix1DVBC{Ws}(A::SparseMatrixCSC{Tv, Ti}, Φ::SplitPartition{Ti}) where {Ws, Tv, Ti}
    @inbounds begin
        (m, n) = size(A)
        hst = zeros(Ti, m + 1)
        A_pos = A.colptr
        A_idx = A.rowval
        L = length(Φ)
        pos = undefs(Ti, L + 1)
        ofs = undefs(Ti, L + 1)
        pos[1] = 1
        ofs[1] = 1
        for l = 1:L
            pos[l + 1] = pos[l]
            j = Φ.spl[l]
            j′ = Φ.spl[l + 1]
            for q = A_pos[j]:A_pos[j′] - 1
                i = A_idx[q]
                pos[l + 1] += (hst[i] < l)
                hst[i] = l
            end
            ofs[l + 1] = ofs[l] + (pos[l + 1] - pos[l]) * (j′ - j)
        end
        return _construct_SparseMatrix1DVBC(Val(Ws), A, Φ, pos, ofs)
    end
end

function _construct_SparseMatrix1DVBC(::Val{Ws}, A::SparseMatrixCSC{Tv, Ti}, Φ::SplitPartition{Ti}, pos::Vector{Ti}, ofs::Vector{Ti}) where {Ws, Tv, Ti}
    @inbounds begin
        # matrix notation...
        # i = 1:m rows, j = 1:n columns
        m, n = size(A)

        A_pos = A.colptr
        A_idx = A.rowval
        A_val = A.nzval

        L = length(Φ)
        spl = Φ.spl

        idx = Vector{Ti}(undef, pos[end] - 1)
        val = Vector{Tv}(undef, ofs[end] - 1 + max(Ws...))
        for Q = ofs[end] : ofs[end]  - 1 + max(Ws...) #extra crap at the end keeps vector access in bounds 
            val[Q] = zero(Tv)
        end

        A_q = ones(Int, max(Ws...))

        for l = 1:L
            j = spl[l]
            w = spl[l + 1] - j
            @assert w <= max(Ws...)
            if w == 1
                Q = pos[l]
                q = ofs[l]
                for A_q_1 = A_pos[j]:(A_pos[j + w] - 1)
                    idx[Q] = A_idx[A_q_1]
                    val[q] = A_val[A_q_1]
                    Q += 1
                    q += 1
                end
            else
                i = m + 1
                for Δj = 1:w
                    A_q[Δj] = A_pos[j + Δj - 1]
                    if A_q[Δj] < A_pos[j + Δj]
                        i = min(i, A_idx[A_q[Δj]])
                    end
                end
                Q = pos[l]
                q = ofs[l]
                while i != m + 1
                    i′ = m + 1
                    for Δj = 1:w
                        if A_q[Δj] < A_pos[j + Δj]
                            if A_idx[A_q[Δj]] == i
                                val[q] = A_val[A_q[Δj]] 
                                A_q[Δj] += 1
                            else
                                val[q] = zero(Tv)
                            end
                            if A_q[Δj] < A_pos[j + Δj]
                                i′ = min(i′, A_idx[A_q[Δj]])
                            end
                        else
                            val[q] = zero(Tv)
                        end
                        q += 1
                    end
                    idx[Q] = i
                    Q += 1
                    i = i′
                end
            end
        end
        return SparseMatrix1DVBC{Ws, Tv, Ti}(m, n, spl, pos, idx, ofs, val)
    end
end

function SparseMatrix1DVBC{Ws}(A::SparseMatrixCSC{Tv, Ti}, method::StrictChunker) where {Ws, Tv, Ti}
    @inbounds begin
        # matrix notation...
        # i = 1:m rows, j = 1:n columns
        m, n = size(A)

        Φ = pack_stripe(A, method)

        A_pos = A.colptr
        A_idx = A.rowval
        A_val = A.nzval

        L = length(Φ)
        spl = Φ.spl
        pos = undefs(Ti, L + 1)
        ofs = undefs(Ti, L + 1)
        pos[1] = 1
        ofs[1] = 1
        for l = 1:L
            j = spl[l]
            j′ = spl[l + 1]
            pos[l + 1] = pos[l] + A_pos[min(j + 1, j′)] - A_pos[j]
            ofs[l + 1] = A_pos[j′]
        end
        idx = Vector{Ti}(undef, pos[end] - 1)
        val = Vector{Tv}(undef, ofs[end] - 1 + max(Ws...))
        for q = ofs[end] : ofs[end]  - 1 + max(Ws...) #extra crap at the end keeps vector access in bounds 
            val[q] = zero(Tv)
        end

        A_q = ones(Int, max(Ws...))

        for l = 1:L
            j = spl[l]
            w = spl[l + 1] - j
            @assert w <= max(Ws...)
            for Q = 0 : A_pos[j + 1] - A_pos[j] - 1
                idx[pos[l] + Q] = A_idx[A_pos[j] + Q]
            end
            for j′ = spl[l] : (spl[l + 1] - 1)
                for Q = 0 : A_pos[j + 1] - A_pos[j] - 1
                    val[ofs[l] + Q * w + j′ - j] = A_val[A_pos[j′] + Q]
                end
            end
        end
        return SparseMatrix1DVBC{Ws, Tv, Ti}(m, n, spl, pos, idx, ofs, val)
    end
end