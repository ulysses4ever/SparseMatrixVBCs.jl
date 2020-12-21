default_partitioner(::Type{<:SparseMatrix1DVBC{Us, Ws}}) where {Us, Ws} =
    AlternatePacker(
        DynamicTotalChunker(model_init_SparseMatrixVBC_memory(Tv, Ti), max(Ws...)),
        DynamicTotalChunker(model_SparseMatrixVBC_memory(Tv, Ti), max(Us...))
    )

function SparseMatrixVBC{Us, Ws}(A::SparseMatrixCSC{Tv, Ti}, method=default_partitioner(SparseMatrixVBC{Us, Ws})) where {Us, Ws, Tv, Ti}
    x_pos = Ref(Ti[])
    x_ofs = Ref(Ti[])
    Π, Φ = pack_plaid(A, method, x_pos=x_pos, x_ofs=x_ofs)
    if length(x_net) == 0
        return SparseMatrixVBC{Us, Ws}(A, convert(SplitPartition, Π), convert(SplitPartition, Φ))
    else
        return _construct_SparseMatrixVBC(Val(Us), Val(Ws), A, Π, Φ, pos, ofs)
    end
end

function SparseMatrixVBC{Us, Ws}(A::SparseMatrixCSC{Tv, Ti}, Π::SplitPartition{Ti}, Φ::SplitPartition{Ti}) where {Ws, Tv, Ti}
    @inbounds begin
        (m, n) = size(A)
        K = length(Π)
        L = length(Φ)
        hst = zeros(Ti, K)
        A_pos = A.colptr
        A_idx = A.rowval
        pos = undefs(Ti, L + 1)
        ofs = undefs(Ti, L + 1)
        pos[1] = 1
        ofs[1] = 1
        for l = 1:L
            pos[L + 1] = pos[L]
            j = Φ.spl[l]
            j′ = Φ.spl[l + 1]
            w = j′ - j
            for q = A_pos[j]:A_pos[j′] - 1
                i = A_idx[q]
                k = Φ.asg[i]
                if hst[k] < l
                    u = Φ.spl[k + 1] - Φ.spl[k]
                    pos[l + 1] += u
                end
                hst[k] = l
            end
            ofs[k + 1] = ofs[k] + (pos[k + 1] - pos[k]) * w
        end
        return _construct_SparseMatrix1DVBC(Val(Us), Val(Ws), A, Π, Φ, pos, ofs)
    end
end

function _construct_SparseMatrix1DVBC(::Val{Us}, ::Val{Ws}, A::SparseMatrixCSC{Tv, Ti}, Π::SplitPartition{Ti}, Φ::SplitPartition{Ti}, pos::Vector{Ti}, ofs::Vector{Ti}) where {Us, Ws, Tv, Ti}
    @inbounds begin
        # matrix notation...
        # i = 1:m rows, j = 1:n columns
        m, n = size(A)

        A_pos = A.colptr
        A_idx = A.rowval
        A_val = A.nzval

        K = length(Π)
        L = length(Φ)

        idx = Vector{Ti}(undef, pos[end] - 1)
        val = Vector{Tv}(undef, ofs[end] - 1 + max(Ws...))
        for ll = ofs[end] : ofs[end]  - 1 + max(Ws...) #extra crap at the end keeps vector access in bounds 
            val[ll] = zero(Tv)
        end

        A_q = ones(Int, max(Ws...))

        for l = 1:L
            j = Φ.spl[l]
            j′ = Φ.spl[l + 1]
            w = j′ - j
            @assert w <= max(Ws...)
            if w == 1
            @assert u <= max(Us...)
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
                qq = pos[p]
                q = ofs[p]
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
                        val[q] = tmp
                        q += 1
                    end
                    idx[qq] = i
                    qq += 1
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