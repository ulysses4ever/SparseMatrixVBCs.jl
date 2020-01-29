struct EminentBlocker end

function blocks(B::SparseMatrixCSC{Tv, Ti}, w_max, method::EminentBlocker) where {Tv, Ti}
    @inbounds begin
        # matrix notation...
        # i = 1:m rows, j = 1:n columns
        m, n = size(B)

        i = rowvals(B)

        hst = zeros(Int, m)

        spl = zeros(Int, n + 1)
        pos = zeros(Int, n + 1)
        ofs = zeros(Int, n + 1)

        d = length(nzrange(B, 1))
        j = 1
        k = 0
        pos[1] = 1
        ofs[1] = 1
        for j′ = 1:n
            Δ = false
            d′ = length(nzrange(B, j′))
            if d′ == d
                for nz in nzrange(B, j′)
                    if hst[i[nz]] < j
                        Δ = true
                    end
                    hst[i[nz]] = j′
                end
            else
                Δ = true
            end
            w = j′ - j + 1
            if Δ || w == w_max + 1
                k += 1
                j = j′
                d = d′
                spl[k] = j
                pos[k + 1] = pos[k] + d
                ofs[k + 1] = ofs[k] + w * d
            end
        end
        spl[k + 1] = n + 1

        resize!(spl, k + 1)
        resize!(pos, k + 1)
        resize!(ofs, k + 1)
        return (spl, pos, ofs)
    end
end