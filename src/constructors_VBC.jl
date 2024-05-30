default_partitioner(::Type{<:SparseMatrixVBC{U, W, Tv, Ti}}) where {U, W, Tv, Ti} =
    AlternatePacker(
        EquiChunker(),
        EquiChunker(),
        DynamicTotalChunker(ConstrainedCost(model_SparseMatrixVBC_memory(Tv, Ti), VertexCount(), W)),
        DynamicTotalChunker(ConstrainedCost(permutedims(model_SparseMatrixVBC_memory(Tv, Ti)), VertexCount(), U)),
        DynamicTotalChunker(ConstrainedCost(model_SparseMatrixVBC_memory(Tv, Ti), VertexCount(), W)),
    )

function SparseMatrixVBC{U, W}(A::SparseMatrixCSC{Tv, Ti}, method=default_partitioner(SparseMatrixVBC{U, W, Tv, Ti})) where {U, W, Tv, Ti}
    Π, Φ = pack_plaid(A, method)
    return SparseMatrixVBC{U, W}(A, convert(SplitPartition, Π), convert(SplitPartition, Φ))
end

function SparseMatrixVBC{U, W}(A::SparseMatrixCSC{Tv, Ti}, Π::SplitPartition{Ti}, Φ::SplitPartition{Ti}) where {U, W, Tv, Ti}
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
        A_val = A.nzval
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

        idx = Vector{Ti}(undef, pos[end] - 1)
        Δw = fld(CpuId.simdbytes(), sizeof(Tv)) #TODO need to define an upper limit on this one
        val = Vector{Tv}(undef, ofs[end] - 1 + U * (Δw * cld(W, Δw)))
        for Q = ofs[end] : ofs[end]  - 1 + U * (Δw * cld(W, Δw)) #extra stuff to keep vector access in bounds 
            val[Q] = zero(Tv)
        end

        A_q = ones(Int, W)

        for k = 1:K
            @assert Π_spl[k + 1] - Π_spl[k] <= U
        end

        for l = 1:L
            j = Φ_spl[l]
            w = Φ_spl[l + 1] - j
            @assert w <= W
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
        return SparseMatrixVBC{U, W, Tv, Ti}(m, n, Π, Φ, pos, idx, ofs, val)
    end
end