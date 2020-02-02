struct NaturalPartitioner end

function partition(A::SparseMatrixCSC{Tv, Ti}, w_max, method::NaturalPartitioner) where {Tv, Ti}
    @inbounds begin
        # matrix notation...
        # i = 1:m rows, j = 1:n columns
        m, n = size(A)

        A_pos = A.colptr
        A_idx = A.rowval

        #hst = zeros(Int, m)

        spl = Vector{Int}(undef, n + 1) # Column split locations
        pos = Vector{Int}(undef, n + 1) # Number of stored indices so far
        ofs = Vector{Int}(undef, n + 1) # Number of stored values so far

        c = A_pos[2] - A_pos[1] #The cardinality of the first column in the part
        j = 1
        k = 0
        spl[1] = 1
        pos[1] = 1
        ofs[1] = 1
        for j′ = 2:n
            c′ = A_pos[j′ + 1] - A_pos[j′] #The cardinality of the candidate column
            w = j′ - j #Current block size
            v_j = @view(A_idx[A_pos[j]:(A_pos[j + 1] - 1)])
            v_j′ = @view(A_idx[A_pos[j′]:(A_pos[j′ + 1] - 1)])
            if w == w_max || v_j != v_j′
                k += 1
                spl[k + 1] = j′
                pos[k + 1] = pos[k] + c
                ofs[k + 1] = ofs[k] + w * c
                j = j′
                c = c′
            end
        end
        j′ = n + 1
        w = j′ - j
        k += 1
        spl[k + 1] = j′
        pos[k + 1] = pos[k] + c
        ofs[k + 1] = ofs[k] + w * c

        resize!(spl, k + 1)
        resize!(pos, k + 1)
        resize!(ofs, k + 1)

        return Partition{Ti}(spl, pos, ofs)
    end
end

function SparseMatrix1DVBC{Ws}(A::SparseMatrixCSC{Tv, Ti}, method::NaturalPartitioner) where {Ws, Tv, Ti}
    @inbounds begin
        # matrix notation...
        # i = 1:m rows, j = 1:n columns
        m, n = size(A)

        B_prt = partition(A, max(Ws...), NaturalPartitioner())

        A_pos = A.colptr
        A_idx = A.rowval
        A_val = A.nzval

        k = length(B_prt)

        spl = B_prt.spl
        pos = B_prt.pos
        idx = Vector{Ti}(undef, pos[end] - 1)
        ofs = B_prt.ofs
        val = Vector{Tv}(undef, ofs[end] - 1 + max(Ws...))
        for qq = ofs[end] : ofs[end]  - 1 + max(Ws...) #extra crap at the end keeps vector access in bounds 
            val[qq] = zero(Tv)
        end

        A_q = ones(Int, max(Ws...))

        for p = 1:k
            j = spl[p]
            w = spl[p + 1] - j
            @assert w <= max(Ws...)
            for q = 0 : A_pos[j + 1] - A_pos[j] - 1
                idx[pos[p] + q] = A_idx[A_pos[j] + q]
            end
            for j′ = spl[p] : (spl[p + 1] - 1)
                for q = 0 : A_pos[j + 1] - A_pos[j] - 1
                    val[ofs[p] + q * w + j′ - j] = A_val[A_pos[j′] + q]
                end
            end
        end
        return SparseMatrix1DVBC{Ws, Tv, Ti}(m, n, spl, pos, idx, ofs, val)
    end
end