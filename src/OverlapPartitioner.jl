struct OverlapPartitioner
    ρ::Float64
end

function partition(A::SparseMatrixCSC{Tv, Ti}, w_max, method::OverlapPartitioner) where {F, Tv, Ti}
    @inbounds begin
        # matrix notation...
        # i = 1:m rows, j = 1:n columns
        m, n = size(A)

        A_pos = A.colptr
        A_idx = A.rowval

        ρ = method.ρ

        hst = zeros(Int, m)

        Π = Vector{Int}(undef, n + 1) # Column split locations
        pos = Vector{Int}(undef, n + 1) # Number of stored indices so far
        ofs = Vector{Int}(undef, n + 1) # Number of stored values so far

        d = A_pos[2] - A_pos[1] #The number of distinct values in the part
        c = A_pos[2] - A_pos[1] #The cardinality of the first column in the part
        j = 1
        K = 0
        Π[1] = 1
        pos[1] = 1
        ofs[1] = 1
        for i in @view A_idx[A_pos[1]:(A_pos[2] - 1)]
            hst[i] = 1
        end
        for j′ = 2:n
            c′ = A_pos[j′ + 1] - A_pos[j′] #The cardinality of the candidate column
            d′ = d #Becomes the number of distinct values in the candidate part
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
                    d′ += 1
                    hst[i] = j′
                end
            end
            w = j′ - j #Current block size
            if w == w_max || cc′ < ρ * min(c, c′)
                K += 1
                Π[K + 1] = j′
                pos[K + 1] = pos[K] + d
                ofs[K + 1] = ofs[K] + w * d
                j = j′
                d = c′
            else
                d = d′
            end
        end
        j′ = n + 1
        w = j′ - j
        K += 1
        Π[K + 1] = j′
        pos[K + 1] = pos[K] + d
        ofs[K + 1] = ofs[K] + w * d

        resize!(Π, K + 1)
        resize!(pos, K + 1)
        resize!(ofs, K + 1)
        return Partition{Ti}(Π, pos, ofs)
    end
end