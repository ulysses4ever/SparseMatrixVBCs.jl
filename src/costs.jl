model_SparseMatrix1DVBC_blocks(W) = AffineNetCostModel(0, 0, 0, 1)

model_SparseMatrix1DVBC_memory(W, Tv, Ti) = ColumnBlockComponentCostModel{Int}(W, 3 * sizeof(Ti), (w) -> sizeof(Ti) + w * sizeof(Tv))

model_SparseMatrix1DVBC_time(W, Tv, Ti) = ColumnBlockComponentCostModel{Float64}(W, model_SparseMatrix1DVBC_time_params(W, Tv, Ti)...)

@memoize @Vault() function model_SparseMatrix1DVBC_time_params(W, Tv, Ti, arch=arch_id())
    @info "calculating $(SparseMatrix1DVBC{W, Tv, Ti}) cost model..."
    @assert arch == arch_id()

    #ms = (i = 1; [1; [i = max(i + 1, i + fld(prevpow(2, i), 4)) for _ = 1:19]])
    ms = [2^i for i = 0:8]
    mem_max = fld(first(filter(t->t.type_==:L1Cache, collect(Hwloc.topology_load()))).attr.size, 2) #Half the L1 cache size. Could be improved.
    T = Float64[]
    ds = Vector{Float64}[]
    C = Float64[]
    for w in 1:W
        ts = Float64[]
        c = 0
        for m in ms
            L = fld(mem_max, 3 * sizeof(Ti) + m * (sizeof(Tv) * w + sizeof(Ti))) #TODO incoroporate size of x and y
            if L >= 1
                n = w * L
                A = sparse(ones(Tv, m, n))
                B = SparseMatrix1DVBC{W}(A, pack_stripe(A, EquiChunker(w)))
                x = ones(m) #TODO add x and y eltype and Δw info to autotuned params
                y = ones(n) #TODO add x and y eltype and Δw info to autotuned params
                d_α_col = zeros(W)
                d_α_col[w] = L
                d_β_col = zeros(W)
                d_β_col[w] = L * m
                d = [d_α_col; d_β_col]
                mul!(y, B', x)
                t = (@belapsed mul!($y, $B', $x) evals=1_000)
                push!(ds, d)
                push!(T, t)
                @info "w: $w m: $m n: $n L: $L t: $t"
                c += 1
            end
        end
        append!(C, [1/c for _ = 1:c])
    end
    D = reduce(hcat, ds)
    println(size(D))
    P = (Diagonal(sqrt.(C)) * D') \ (Diagonal(sqrt.(C)) * T)
    α_col = (P[1:W]...,)
    β_col = (P[W + 1:end]...,)
    @info "α_col: $α_col"
    @info "β_col: $β_col"
    @info "done!"
    return (α_col, β_col)
end

model_SparseMatrixVBC_blocks(U, W) = BlockComponentCostModel{Int}(U, W, 0, 0, (1,), (1, ))

model_SparseMatrixVBC_memory(U, W, Tv, Ti) = BlockComponentCostModel{Int}(U, W, sizeof(Ti), 3 * sizeof(Ti), (1, identity), (sizeof(Ti), (w)->(sizeof(Tv) * w)))

model_SparseMatrixVBC_time(R, U, W, Tv, Ti) = BlockComponentCostModel{Float64}(U, W, model_SparseMatrixVBC_time_params(R, U, W, Tv, Ti)...)

@memoize @Vault() function model_SparseMatrixVBC_time_params(R, U, W, Tv, Ti, arch=arch_id())
    @info "calculating $(SparseMatrixVBC{U, W, Tv, Ti}) cost model..."
    @assert arch == arch_id()

    #Ks = (i = 1; [1; [i = max(i + 1, i + fld(prevpow(4, i), 2)) for _ = 1:8]])
    Ks = [4^i for i = 0:8]
    Ls = Ks
    mem_max = fld(first(filter(t->t.type_==:L1Cache, collect(Hwloc.topology_load()))).attr.size, 2) #Half the L1 cache size. Could be improved.
    T = Float64[]
    ds = Vector{Float64}[]
    C = Float64[]
    for u in 1:U
        for w in 1:W
            ts = Float64[]
            c = 0
            for K in Ks
                for L in Ls
                    if u * K * w * L * sizeof(Tv) + L * K * sizeof(Ti) <= mem_max
                        (m, n) = (u * K, w * L)
                        A = sparse(ones(Tv, m, n))
                        B = SparseMatrixVBC{U, W}(A, pack_stripe(A', EquiChunker(u)), pack_stripe(A, EquiChunker(w)))
                        x = ones(m) #TODO add x and y eltype and Δw info to autotuned params
                        y = ones(n) #TODO add x and y eltype and Δw info to autotuned params
                        if Base.summarysize(B) + Base.summarysize(x) + Base.summarysize(y) < mem_max
                            d_α_row = zeros(U)
                            d_α_row[u] = K
                            d_α_col = zeros(W)
                            d_α_col[w] = L
                            d_β = zeros(U, W)
                            d_β[u, w] = L * K
                            d = [d_α_row; d_α_col; reshape(d_β, :)]
                            mul!(y, B', x)
                            t = (@belapsed mul!($y, $B', $x) evals=1_000)
                            push!(ds, d)
                            push!(T, t)
                            @info "u: $u w: $w m: $m n: $n K: $K L: $L t: $t"
                            c += 1
                        end
                    end
                end
            end
            append!(C, [1/c for _ = 1:c])
        end
    end
    D = reduce(hcat, ds)
    P = (Diagonal(sqrt.(C)) * D') \ (Diagonal(sqrt.(C)) * T)
    α_row = (P[1:U]...,)
    α_col = (P[U + 1:U + W]...,)
    β = collect(reshape(P[U + W + 1:end], U, W))
    F = svd(β)
    β_row = (((F.U[:,r]...,) for r = 1:R)...,)
    β_col = ((((F.S[r] * F.V[:,r])...,) for r = 1:R)...,)
    @info "α_row: $α_row"
    @info "α_col: $α_col"
    @info "β: $β"
    @info "β_reconstruct: $([sum(β_row[r][u] * β_col[r][w] for r = 1:R) for u = 1:U, w = 1:W])"
    @info "β_row: $β_row"
    @info "β_col: $β_col"
    @info "done!"
    return (α_row, α_col, β_row, β_col)
end