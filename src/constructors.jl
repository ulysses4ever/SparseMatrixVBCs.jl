function SparseMatrix1DVBC{Ws}(A::SparseMatrixCSC{Tv, Ti}, method=OptimalPartitioner(VBCCSCMemoryCost{Tv, Ti}())) where {Ws, Tv, Ti}
    SparseMatrix1DVBC{Ws}(A, partition(A, max(Ws...), method))
end

function SparseMatrix1DVBC{Ws}(A::SparseMatrixCSC{Tv, Ti}, B_prt::Partition{Ti}) where {Ws, Tv, Ti}
    @inbounds begin
        # matrix notation...
        # i = 1:m rows, j = 1:n columns
        m, n = size(A)


        A_pos = A.colptr
        A_idx = A.rowval
        A_val = A.nzval

        k = length(B_prt)

        Π = B_prt.Π
        pos = B_prt.pos
        idx = Vector{Ti}(undef, pos[end] - 1)
        ofs = B_prt.ofs
        val = Vector{Tv}(undef, ofs[end] - 1 + max(Ws...))
        for rr = ofs[end] : ofs[end]  - 1 + max(Ws...) #extra crap at the end keeps vector access in bounds 
            val[rr] = zero(Tv)
        end

        A_q = ones(Int, max(Ws...))

        for p = 1:k
            j = Π[p]
            w = Π[p + 1] - j
            @assert w <= max(Ws...)
            if w == 1
                rr = pos[p]
                r = ofs[p]
                for A_q_1 = A_pos[j]:(A_pos[j + w] - 1)
                    idx[rr] = A_idx[A_q_1]
                    val[r] = A_val[A_q_1]
                    rr += 1
                    r += 1
                end
            else
                i = m + 1
                for Δj = 1:w
                    A_q[Δj] = A_pos[j + Δj - 1]
                    if A_q[Δj] < A_pos[j + Δj]
                        i = min(i, A_idx[A_q[Δj]])
                    end
                end
                rr = pos[p]
                r = ofs[p]
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
                        val[r] = tmp
                        r += 1
                    end
                    idx[rr] = i
                    rr += 1
                    i = i′
                end
            end
        end
        return SparseMatrix1DVBC{Ws, Tv, Ti}(m, n, Π, pos, idx, ofs, val)
    end
end