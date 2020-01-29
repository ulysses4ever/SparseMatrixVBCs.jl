struct EminentPartitioner end

function partition(A::SparseMatrixCSC{Tv, Ti}, w_max, method::EminentPartitioner) where {Tv, Ti}
    begin
        # matrix notation...
        # i = 1:m rows, j = 1:n columns
        m, n = size(A)

        A_pos = A.colptr
        A_idx = A.rowval

        hst = zeros(Int, m)

        spl = Vector{Int}(undef, n + 1) # Column split locations
        pos = Vector{Int}(undef, n + 1) # Number of stored indices so far
        ofs = Vector{Int}(undef, n + 1) # Number of stored values so far

        c = A_pos[2] - A_pos[1] #The cardinality of the first column in the part
        j = 1
        k = 0
        spl[1] = 1
        pos[1] = 1
        ofs[1] = 1
        for i in @view A_idx[A_pos[1]:(A_pos[2] - 1)]
            hst[i] = 1
        end
        for j′ = 2:n
            c′ = A_pos[j′ + 1] - A_pos[j′] #The cardinality of the candidate column
            cc′ = 0 #The cardinality of the intersection between column j and j′
            for i in @view A_idx[A_pos[j′]:(A_pos[j′ + 1] - 1)]
                h = hst[i]
                if abs(h) == j
                    cc′ += 1
                    hst[i] = -j′
                elseif j < h
                    hst[i] = j′
                elseif h < -j
                    cc′ += 1
                    hst[i] = -j′
                else
                    hst[i] = j′
                end
            end
            w = j′ - j #Current block size
            if w == w_max || cc′ != c || c != c′
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