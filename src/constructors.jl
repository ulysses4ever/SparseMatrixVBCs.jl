function SparseMatrix1DVBC{Ws}(A::SparseMatrixCSC{Tv, Ti}, method=DynamicTotalChunker(model_SparseMatrix1DVBC_memory(Tv, Ti), max(Ws...))) where {Ws, Tv, Ti}
    x_net = Ref(Ti[])
    Φ = pack_stripe(A, method, x_net=x_net)
    if length(x_net) == 0
        return SparseMatrix1DVBC{Ws}(A, Φ)
    else
        K = length(Φ)
        Φ = convert(SplitPartition, Φ)
        spl = Φ.spl
        pos = undefs(Ti, K + 1)
        ofs = undefs(Ti, K + 1)
        pos[1] = 1
        ofs[1] = 1
        for k = 1:K
            pos[k + 1] = pos[k] + x_net[][k]
            ofs[k + 1] = ofs[k] + x_net[][k] * (spl[k + 1] - spl[k])
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
        K = length(Φ)
        pos = undefs(Ti, K + 1)
        ofs = undefs(Ti, K + 1)
        pos[1] = 1
        ofs[1] = 1
        for k = 1:K
            pos[k + 1] = pos[k]
            j = Φ.spl[k]
            j′ = Φ.spl[k + 1]
            for q = A_pos[j]:A_pos[j′] - 1
                i = A_idx[q]
                pos[k + 1] += (hst[i] < k)
                hst[i] = k
            end
            ofs[k + 1] = ofs[k] + (pos[k + 1] - pos[k]) * (j′ - j)
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

        K = length(Φ)
        spl = Φ.spl

        idx = Vector{Ti}(undef, pos[end] - 1)
        val = Vector{Tv}(undef, ofs[end] - 1 + max(Ws...))
        for ll = ofs[end] : ofs[end]  - 1 + max(Ws...) #extra crap at the end keeps vector access in bounds 
            val[ll] = zero(Tv)
        end

        A_q = ones(Int, max(Ws...))

        for p = 1:K
            j = spl[p]
            w = spl[p + 1] - j
            @assert w <= max(Ws...)
            if w == 1
                ll = pos[p]
                l = ofs[p]
                for A_q_1 = A_pos[j]:(A_pos[j + w] - 1)
                    idx[ll] = A_idx[A_q_1]
                    val[l] = A_val[A_q_1]
                    ll += 1
                    l += 1
                end
            else
                i = m + 1
                for Δj = 1:w
                    A_q[Δj] = A_pos[j + Δj - 1]
                    if A_q[Δj] < A_pos[j + Δj]
                        i = min(i, A_idx[A_q[Δj]])
                    end
                end
                ll = pos[p]
                l = ofs[p]
                while i != m + 1
                    i′ = m + 1
                    for Δj = 1:w
                        tmp = zero(Tv)
                        if A_q[Δj] < A_pos[j + Δj]
                            if A_idx[A_q[Δj]] == i
                                tmp = A_val[A_q[Δj]] 
                                A_q[Δj] += 1
                            end
                            if A_q[Δj] < A_pos[j + Δj]
                                i′ = min(i′, A_idx[A_q[Δj]])
                            end
                        end
                        val[l] = tmp
                        l += 1
                    end
                    idx[ll] = i
                    ll += 1
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

        K = length(Φ)
        spl = Φ.spl
        pos = undefs(Ti, K + 1)
        ofs = undefs(Ti, K + 1)
        pos[1] = 1
        ofs[1] = 1
        for k = 1:K
            j = spl[k]
            j′ = spl[k + 1]
            pos[k + 1] = pos[k] + A_pos[min(j + 1, j′)] - A_pos[j]
            ofs[k + 1] = A_pos[j′]
        end
        idx = Vector{Ti}(undef, pos[end] - 1)
        val = Vector{Tv}(undef, ofs[end] - 1 + max(Ws...))
        for ll = ofs[end] : ofs[end]  - 1 + max(Ws...) #extra crap at the end keeps vector access in bounds 
            val[ll] = zero(Tv)
        end

        A_q = ones(Int, max(Ws...))

        for p = 1:K
            j = spl[p]
            w = spl[p + 1] - j
            @assert w <= max(Ws...)
            for l = 0 : A_pos[j + 1] - A_pos[j] - 1
                idx[pos[p] + l] = A_idx[A_pos[j] + l]
            end
            for j′ = spl[p] : (spl[p + 1] - 1)
                for l = 0 : A_pos[j + 1] - A_pos[j] - 1
                    val[ofs[p] + l * w + j′ - j] = A_val[A_pos[j′] + l]
                end
            end
        end
        return SparseMatrix1DVBC{Ws, Tv, Ti}(m, n, spl, pos, idx, ofs, val)
    end
end