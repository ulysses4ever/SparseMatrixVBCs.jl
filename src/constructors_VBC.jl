default_partitioner(::Type{<:SparseMatrix1DVBC{U_max, Ws}}) where {U_max, Ws} =
    AlternatePacker(
        DynamicTotalChunker(model_init_SparseMatrixVBC_memory(Tv, Ti), max(Ws...)),
        DynamicTotalChunker(model_SparseMatrixVBC_memory(Tv, Ti), U_max)
    )

function SparseMatrixVBC{U_max, Ws}(A::SparseMatrixCSC{Tv, Ti}, method=default_partitioner(SparseMatrixVBC{U_max, Ws})) where {U_max, Ws, Tv, Ti}
    x_pos = Ref(Ti[])
    x_ofs = Ref(Ti[])
    Π, Φ = pack_plaid(A, method, x_pos=x_pos, x_ofs=x_ofs)
    if length(x_pos[]) == 0 || length(x_ofs[]) == 0
        return SparseMatrixVBC{U_max, Ws}(A, convert(SplitPartition, Π), convert(SplitPartition, Φ))
    else
        return _construct_SparseMatrixVBC(Val(U_max), Val(Ws), A, Π, Φ, x_pos[], x_ofs[])
    end
end

function SparseMatrixVBC{U_max, Ws}(A::SparseMatrixCSC{Tv, Ti}, Π::SplitPartition{Ti}, Φ::SplitPartition{Ti}) where {U_max, Ws, Tv, Ti}
    @inbounds begin
        (m, n) = size(A)
        K = length(Π)
        L = length(Φ)
        Π_spl = Π.spl
        Φ_spl = Φ.spl
        Π_asg = convert(MapPartition, Π).asg
        hst = zeros(Ti, K)
        A_pos = A.colptr
        A_idx = A.rowval
        pos = undefs(Ti, L + 1)
        ofs = undefs(Ti, L + 1)
        pos[1] = 1
        ofs[1] = 1
        for l = 1:L
            ofs[l + 1] = ofs[l]
            pos[l + 1] = pos[l]
            j = Φ_spl[l]
            j′ = Φ_spl[l + 1]
            w = j′ - j
            for q = A_pos[j]:A_pos[j′] - 1
                i = A_idx[q]
                k = Π_asg[i]
                if hst[k] < l
                    u = Π_spl[k + 1] - Π_spl[k]
                    pos[l + 1] += 1
                    ofs[l + 1] += u * w
                end
                hst[k] = l
            end
        end
        return _construct_SparseMatrixVBC(Val(U_max), Val(Ws), A, Π, Φ, Π_asg, pos, ofs)
    end
end

function _construct_SparseMatrixVBC(::Val{U_max}, ::Val{Ws}, A::SparseMatrixCSC{Tv, Ti}, Π::SplitPartition{Ti}, Φ::SplitPartition{Ti}, Π_asg::Vector{Ti}, pos::Vector{Ti}, ofs::Vector{Ti}) where {U_max, Ws, Tv, Ti}
    @inbounds begin
        # matrix notation...
        # i = 1:m rows, j = 1:n columns
        m, n = size(A)

        A_pos = A.colptr
        A_idx = A.rowval
        A_val = A.nzval

        K = length(Π)
        L = length(Φ)
        Π_spl = Π.spl
        Φ_spl = Φ.spl

        idx = Vector{Ti}(undef, pos[end] - 1)
        val = Vector{Tv}(undef, ofs[end] - 1 + U_max * max(Ws...))
        for Q = ofs[end] : ofs[end]  - 1 + U_max * max(Ws...) #extra crap at the end keeps vector access in bounds 
            val[Q] = zero(Tv)
        end

        A_q = ones(Int, max(Ws...))

        for k = 1:K
            @assert Π_spl[k + 1] - Π_spl[k] <= U_max
        end

        for l = 1:L
            j = Φ_spl[l]
            w = Φ_spl[l + 1] - j
            @assert w <= max(Ws...)
            if w == 1
                Q = pos[l]
                q = ofs[l]
                A_q_1 = A_pos[j]
                while A_q_1 < A_pos[j + 1]
                    k = Π_asg[A_idx[A_q_1]]
                    for i = Π_spl[k] : Π_spl[k + 1] - 1
                        if A_q_1 < A_pos[j + 1] && A_idx[A_q_1] == i
                            val[q] = A_val[A_q_1]
                            A_q_1 += 1
                        else
                            val[q] = zero(Tv)
                        end
                        q += 1
                    end
                    idx[Q] = k
                    Q += 1
                end
            else
                k = K + 1
                for Δj = 1:w
                    A_q[Δj] = A_pos[j + Δj - 1]
                    if A_q[Δj] < A_pos[j + Δj]
                        k = min(k, Π_asg[A_idx[A_q[Δj]]])
                    end
                end
                Q = pos[l]
                q = ofs[l]
                while k != K + 1
                    for i = Π_spl[k] : Π_spl[k + 1] - 2
                        for Δj = 1:w
                            if A_q[Δj] < A_pos[j + Δj] && A_idx[A_q[Δj]] == i
                                val[q] = A_val[A_q[Δj]] 
                                A_q[Δj] += 1
                            else
                                val[q] = zero(Tv)
                            end
                            q += 1
                        end
                    end
                    k′ = K + 1
                    let i = Π_spl[k + 1] - 1
                        for Δj = 1:w
                            if A_q[Δj] < A_pos[j + Δj]
                                if A_idx[A_q[Δj]] == i
                                    val[q] = A_val[A_q[Δj]] 
                                    A_q[Δj] += 1
                                else
                                    val[q] = zero(Tv)
                                end
                                if A_q[Δj] < A_pos[j + Δj]
                                    k′ = min(k′, Π_asg[A_idx[A_q[Δj]]])
                                end
                            else
                                val[q] = zero(Tv)
                            end
                            q += 1
                        end
                    end
                    idx[Q] = k
                    Q += 1
                    k = k′
                end
            end
        end
        return SparseMatrixVBC{U_max, Ws, Tv, Ti}(m, n, Π, Φ, pos, idx, ofs, val)
    end
end