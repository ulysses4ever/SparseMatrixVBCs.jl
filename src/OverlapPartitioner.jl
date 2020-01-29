struct OverlapPartitioner
    ρ::Float64
end

function partition(A::SparseMatrixCSC{Tv, Ti}, w_max, method::OSKIPartitioner) where {F, Tv, Ti}
    @inbounds begin
        # matrix notation...
        # i = 1:m rows, j = 1:n columns
        m, n = size(A)

        A_pos = A.colptr
        A_idx = A.rowval

        hst = zeros(Int, m)

        spl = Vector{Int}(undef, n + 1) # Column split locations
        pos = Vector{Int}(undef, n + 1) # Number of stored indices so far
        ofs = Vector{Int}(undef, n + 1) # Number of stored values so far

        d = A_pos[2] - A_pos[1] #The number of distinct values in the part
        c = A_pos[2] - A_pos[1] #The cardinality of the first column in the part
        j = 1
        k = 0
        spl[1] = 1
        pos[1] = 1
        ofs[1] = 1
        for i in @view A_idx[A.pos[1]:(A_pos[2] - 1)]
            hst[i] = 1
        end
        for j′ = 2:n
            c′ = A_pos[j′ + 1] - A_pos[j′] #The cardinality of the candidate column
            d′ = d #Becomes the number of distinct values in the candidate part
            cc′ = 0 #The cardinality of the intersection between column j and j′
            for i in @view A_idx[A_pos[j′]:(A_pos[j′ + 1] - 1)]
                if abs(hst[i]) == j
                    cc′ += 1
                    hst[i] = -j
                elseif j < hst[i]
                    hst[i] = j
                elseif hst[i] < -j
                    cc′ += 1
                    hst[i] = -j
                else
                    d′ += 1
                    hst[i] = j
                end
            end
            w = j′ - j #Current block size
            if w == w_max || cc′ < method.ρ * min(c, c′)
                k += 1
                spl[k + 1] = j′
                pos[k + 1] = pos[k] + d
                ofs[k + 1] = ofs[k] + w * d
                j = j′
                d = c′
            else
                d = d′
            end
        end
        j′ = n + 1
        w = j′ - j
        k += 1
        spl[k + 1] = j′
        pos[k + 1] = pos[k] + d
        ofs[k + 1] = ofs[k] + w * d

        resize!(spl, k + 1)
        resize!(pos, k + 1)
        resize!(ofs, k + 1)
        return Partition{Ti}(spl, pos, ofs)
    end
end