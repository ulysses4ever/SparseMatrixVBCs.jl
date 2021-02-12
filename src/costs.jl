struct Line{Tv}
    a::Tv
    b::Tv
end

@inline (p::Line)(x) = p.a + p.b * x

model_SparseMatrix1DVBC_blocks() = AffineNetCostModel(0, 0, 0, 1)

model_SparseMatrix1DVBC_memory(Tv, Ti) = ColumnBlockComponentCostModel{Int}(3 * sizeof(Ti), Line(sizeof(Ti), sizeof(Tv)))

model_SparseMatrix1DVBC_TrSpMV_time(W, Tv, Ti, Tu) = ColumnBlockComponentCostModel{Float64}(model_SparseMatrix1DVBC_TrSpMV_time_params(W, Tv, Ti, Tu)...)

@memoize DiskCache(@get_scratch!("1DVBC_TrSpMV_timings")) function model_SparseMatrix1DVBC_TrSpMV_time_params(W, Tv, Ti, Tu, arch=arch_id())
    @info "calculating cost model for $(SparseMatrix1DVBC{W, Tv, Ti})' * $(Vector{Tu})..."
    @assert arch == arch_id()

    #ms = (i = 1; [1; [i = max(i + 1, i + fld(prevpow(2, i), 4)) for _ = 1:19]])
    ms = [2^i for i = 0:8]
    #ms = [16:16:256]
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
                x = ones(Tu, m)
                y = ones(Tu, n)
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
        append!(C, [c for _ = 1:c])
    end
    D = reduce(hcat, ds)
    P = (Diagonal(1 ./ (sqrt.(C) .* T)) * D') \ (1 ./ sqrt.(C))
    α_col = (P[1:W]...,)
    β_col = (P[W + 1:end]...,)
    @info "α_col: $α_col"
    @info "β_col: $β_col"
    @info "done!"
    return (α_col, β_col)
end

model_SparseMatrixVBC_blocks() = BlockComponentCostModel{Int}(0, 0, (1,), (1, ))

model_SparseMatrixVBC_memory(Tv, Ti) = BlockComponentCostModel{Int}(sizeof(Ti), 3 * sizeof(Ti), (Line(1, 0), Line(0, 1)), (Line(sizeof(Ti), 0), Line(0, sizeof(Tv))))

model_SparseMatrixVBC_TrSpMV_time(R, U, W, Tv, Ti, Tu) = BlockComponentCostModel{Float64}(model_SparseMatrixVBC_TrSpMV_time_params(R, U, W, Tv, Ti, Tu)...)

@memoize DiskCache(@get_scratch!("VBC_TrSpMV_timings")) function model_SparseMatrixVBC_TrSpMV_time_params(R, U, W, Tv, Ti, Tu, arch=arch_id())
    @info "calculating cost model for $(SparseMatrixVBC{U, W, Tv, Ti})' * $(Vector{Tu})..."
    @assert arch == arch_id()

    #Ks = (i = 1; [1; [i = max(i + 1, i + fld(prevpow(4, i), 2)) for _ = 1:8]])
    Ks = [2^i for i = 0:8]
    #Ks = [16:16:256]
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
                        x = ones(Tu, m)
                        y = ones(Tu, n)
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
            append!(C, [c for _ = 1:c])
        end
    end
    D = reduce(hcat, ds)
    P = (Diagonal(1 ./ (sqrt.(C) .* T)) * D') \ (1 ./ sqrt.(C))
    α_row = (P[1:U]...,)
    α_col = (P[U + 1:U + W]...,)
    β = collect(reshape(P[U + W + 1:end], U, W))
    F = svd(β)
    β_row = (((F.U[:,r]...,) for r = 1:R)...,)
    β_col = ((((F.S[r] * F.V[:,r])...,) for r = 1:R)...,)
    @info "α_row: $α_row"
    @info "α_col: $α_col"
    @info "β: $β"
    β_reconstruct = [sum(β_row[r][u] * β_col[r][w] for r = 1:R) for u = 1:U, w = 1:W]
    @info "β_reconstruct: $(β_reconstruct)"
    @info "β_error: $(maximum((β_reconstruct .- β)./β))"
    @info "β_row: $β_row"
    @info "β_col: $β_col"
    @info "done!"
    return (α_row, α_col, β_row, β_col)
end