Base.size(mat::SparseMatrix1DVBC) = (mat.m, mat.n)

function SparseMatrix1DVBC{Ws}(arg::SparseMatrixCSC{Tv, Ti}, method=OptimalBlocker(VBCCSCMemoryCost{Tv, Ti}())) where {Ws, Tv, Ti}
    SparseMatrix1DVBC{Ws}(arg, blocks(arg, max(Ws...), method))
end

function SparseMatrix1DVBC{Ws}(arg::SparseMatrixCSC{Tv, Ti}, b::Blocks{Ti}) where {Ws, Tv, Ti}
    @inbounds begin
        # matrix notation...
        # i = 1:m rows, j = 1:n columns
        m, n = size(arg)

        idx = Vector{Ti}(undef, b.pos[end] - 1)
        val = Vector{Tv}(undef, b.ofs[end] - 1 + max(Ws...))
        for qq = b.ofs[end] : b.ofs[end]  - 1 + max(Ws...) #extra crap at the end keeps vector access in bounds 
            val[qq] = zero(Tv)
        end

        arg_q = ones(Int, max(Ws...))

        k = length(b)

        for p = 1:k
            j = b.spl[p]
            w = b.spl[p + 1] - j
            @assert w <= max(Ws...)
            i = m + 1
            for Δj = 1:w
                arg_q[Δj] = arg.colptr[j + Δj - 1]
                if arg_q[Δj] < arg.colptr[j + Δj]
                    i = min(i, arg.rowval[arg_q[Δj]])
                end
            end
            qq = b.pos[p]
            q = b.ofs[p]
            while i != m + 1
                i′ = m + 1
                for Δj = 1:w
                    tmp = zero(Tv)
                    if arg_q[Δj] < arg.colptr[j + Δj]
                        if arg.rowval[arg_q[Δj]] == i
                            tmp = arg.nzval[arg_q[Δj]] 
                            arg_q[Δj] += 1
                        end
                        if arg_q[Δj] < arg.colptr[j + Δj]
                            i′ = min(i′, arg.rowval[arg_q[Δj]])
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
        return SparseMatrix1DVBC{Ws, Tv, Ti}(m, n, b.spl, b.pos, idx, b.ofs, val)
    end
end