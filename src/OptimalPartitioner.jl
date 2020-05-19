struct OptimalPartitioner{G}
    g::G
end

function partition(A::SparseMatrixCSC{Tv, Ti}, w_max, method::OptimalPartitioner{G}) where {G, Tv, Ti}
    @inbounds begin
        # matrix notation...
        # i = 1:m rows, j = 1:n columns
        m, n = size(A)

        A_pos = A.colptr
        A_idx = A.rowval

        g = method.g

        Δ = zeros(Int, n + 1) # Δ is the number of additional distinct entries we see as our part size grows.
        hst = fill(n + 1, m) # hst is the last time we saw some nonzero
        cst = Vector{typeof(zero(g))}(undef, n + 1) # cst[j] is the best cost of a partition from j to n
        dsc = Vector{Int}(undef, n) # dsc[j] is the corresponding number of distinct nonzero entries in the part
        Π = Vector{Int}(undef, n + 1)

        Δ[n + 1] = 0
        cst[n + 1] = zero(g)
        for j = n:-1:1
            d = A_pos[j + 1] - A_pos[j] # The number of distinct nonzero blocks in each candidate part
            Δ[j] = d
            for i in @view A_idx[A_pos[j] : (A_pos[j + 1] - 1)]
                j′ = hst[i]
                if j′ <= j + w_max - 1
                    Δ[j′] -= 1
                end
                hst[i] = j
            end
            best_j′ = j
            best_c = cst[j + 1] + g(1, d)
            best_d = d
            for j′ = j + 1 : min(j + w_max - 1, n)
                d += Δ[j′]
                c = cst[j′ + 1] + g(j′ - j + 1, d) 
                if c < best_c
                    best_c = c
                    best_d = d
                    best_j′ = j′
                end
            end
            cst[j] = best_c
            dsc[j] = best_d
            Π[j] = best_j′
        end

        pos = Vector{Int}(undef, n + 1)
        pos[1] = 1
        ofs = Vector{Int}(undef, n + 1)
        ofs[1] = 1
        k = 0
        j = 1
        while j != n + 1
            j′ = Π[j]
            w = j′ - j + 1
            k += 1
            Π[k] = j
            pos[k + 1] = pos[k] + dsc[j]
            ofs[k + 1] = ofs[k] + w * dsc[j]
            j += w
        end
        Π[k + 1] = j
        resize!(Π, k + 1)
        resize!(pos, k + 1)
        resize!(ofs, k + 1)
        return Partition{Ti}(Π, pos, ofs)
    end
end