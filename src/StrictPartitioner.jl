struct StrictPartitioner end

function partition(A::SparseMatrixCSC{Tv, Ti}, w_max, method::StrictPartitioner) where {Tv, Ti}
    @inbounds begin
        # matrix notation...
        # i = 1:m rows, j = 1:n columns
        m, n = size(A)

        A_pos = A.colptr
        A_idx = A.rowval

        #hst = zeros(Int, m)

        Π = Vector{Int}(undef, n + 1) # Column split locations
        pos = Vector{Int}(undef, n + 1) # Number of stored indices so far
        ofs = Vector{Int}(undef, n + 1) # Number of stored values so far

        c = A_pos[2] - A_pos[1] #The cardinality of the first column in the part
        j = 1
        K = 0
        Π[1] = 1
        pos[1] = 1
        ofs[1] = 1
        for j′ = 2:n
            c′ = A_pos[j′ + 1] - A_pos[j′] #The cardinality of the candidate column
            w = j′ - j #Current block size
            d = true
            if c == c′ && w != w_max
                l′ = A_pos[j′]
                for l = A_pos[j]:(A_pos[j + 1] - 1)
                    if A_idx[l] != A_idx[l′]
                        d = false
                        break
                    end
                    l′ += 1
                end
            else
                d = false
            end
            if !d
                K += 1
                Π[K + 1] = j′
                pos[K + 1] = pos[K] + c
                ofs[K + 1] = ofs[K] + w * c
                j = j′
                c = c′
            end
        end
        j′ = n + 1
        w = j′ - j
        K += 1
        Π[K + 1] = j′
        pos[K + 1] = pos[K] + c
        ofs[K + 1] = ofs[K] + w * c

        resize!(Π, K + 1)
        resize!(pos, K + 1)
        resize!(ofs, K + 1)

        return Partition{Ti}(Π, pos, ofs)
    end
end

function SparseMatrix1DVBC{Ws}(A::SparseMatrixCSC{Tv, Ti}, method::StrictPartitioner) where {Ws, Tv, Ti}
    @inbounds begin
        # matrix notation...
        # i = 1:m rows, j = 1:n columns
        m, n = size(A)

        B_prt = partition(A, max(Ws...), StrictPartitioner())

        A_pos = A.colptr
        A_idx = A.rowval
        A_val = A.nzval

        K = length(B_prt)

        Π = B_prt.Π
        pos = B_prt.pos
        idx = Vector{Ti}(undef, pos[end] - 1)
        ofs = B_prt.ofs
        val = Vector{Tv}(undef, ofs[end] - 1 + max(Ws...))
        for ll = ofs[end] : ofs[end]  - 1 + max(Ws...) #extra crap at the end keeps vector access in bounds 
            val[ll] = zero(Tv)
        end

        A_q = ones(Int, max(Ws...))

        for p = 1:K
            j = Π[p]
            w = Π[p + 1] - j
            @assert w <= max(Ws...)
            for l = 0 : A_pos[j + 1] - A_pos[j] - 1
                idx[pos[p] + l] = A_idx[A_pos[j] + l]
            end
            for j′ = Π[p] : (Π[p + 1] - 1)
                for l = 0 : A_pos[j + 1] - A_pos[j] - 1
                    val[ofs[p] + l * w + j′ - j] = A_val[A_pos[j′] + l]
                end
            end
        end
        return SparseMatrix1DVBC{Ws, Tv, Ti}(m, n, Π, pos, idx, ofs, val)
    end
end