struct Line{Tv}
    a::Tv
    b::Tv
end

@inline (p::Line)(x) = p.a + p.b * x

model_SparseMatrix1DVBC_blocks() = AffineNetCostModel(0, 0, 0, 1)

model_SparseMatrix1DVBC_memory(Tv, Ti) = ColumnBlockComponentCostModel{Int}(3 * sizeof(Ti), Line(sizeof(Ti), sizeof(Tv)))

model_SparseMatrix1DVBC_TrSpMV_time(W, Tv, Ti, Tu) = ColumnBlockComponentCostModel{Float64}(model_SparseMatrix1DVBC_TrSpMV_time_params(W, Tv, Ti, Tu)...)

@memoize DiskCache(joinpath(@get_scratch!("autotune"), "1DVBC_TrSpMV_timings")) function model_SparseMatrix1DVBC_TrSpMV_time_data(W, Tv, Ti, Tu, arch=arch_id())
    @info "collecting data for $(SparseMatrix1DVBC{W, Tv, Ti})' * $(Vector{Tu})..."
    @assert arch == arch_id()

    m₀ = ceil(Int, sqrt(first(filter(t->t.type_==:L1Cache, collect(Hwloc.topology_load()))).attr.size)/sizeof(Tv)) #Fill the L1 cache.
    n₀ = m₀
    T = Float64[]
    ms = Int[]
    ns = Int[]
    Ls = Int[]
    ws = Int[]
    for w in 1:W
        ts = Float64[]
        L₀ = cld(n₀, w)
        for (m, L) in ((m₀, L₀), (m₀, 2L₀), (2m₀, L₀))
            n = w * L
            A = sparse(ones(Tv, m, n))
            B = SparseMatrix1DVBC{W}(A, pack_stripe(A, EquiChunker(w)))
            x = ones(Tu, m)
            y = ones(Tu, n)
            mul!(y, B', x)
            t = (@belapsed mul!($y, $B', $x) evals=1_000)
            push!(ms, m)
            push!(ns, n)
            push!(Ls, L)
            push!(ws, w)
            push!(T, t)
            @info "w: $w m: $m n: $n L: $L t: $t"
        end
    end
    return (ms, ns, Ls, ws, T)
end

@memoize DiskCache(joinpath(@get_scratch!("autotune"), "1DVBC_TrSpMV_time_params")) function model_SparseMatrix1DVBC_TrSpMV_time_params(W, Tv, Ti, Tu, arch=arch_id())
    @info "calculating cost model for $(SparseMatrix1DVBC{W, Tv, Ti})' * $(Vector{Tu})..."
    (ms, ns, Ls, ws, T) = model_SparseMatrix1DVBC_TrSpMV_time_data(W, Tv, Ti, Tu, arch)
    D = hcat(ms, Ls, ns, ms .* Ls, ms .* ns)
    (αβ_row, α_col, β_col, α_block, β_block) = qr(Diagonal(1 ./ T) * D, Val(true)) \ ones(length(T))
    @info "results" (αβ_row, α_col, β_col, α_block, β_block)
    @info "done!"
    return (0.0, Line(α_block, β_block))
end

model_SparseMatrixVBC_blocks() = BlockComponentCostModel{Int}(0, 0, (1,), (1, ))

model_SparseMatrixVBC_memory(Tv, Ti) = BlockComponentCostModel{Int}(sizeof(Ti), 3 * sizeof(Ti), (Line(1, 0), Line(0, 1)), (Line(sizeof(Ti), 0), Line(0, sizeof(Tv))))

model_SparseMatrixVBC_TrSpMV_time(U, W, Tv, Ti, Tu) = BlockComponentCostModel{Float64}(model_SparseMatrixVBC_TrSpMV_time_params(U, W, Tv, Ti, Tu)...)

@memoize DiskCache(joinpath(@get_scratch!("autotune"), "VBC_TrSpMV_timings")) function model_SparseMatrixVBC_TrSpMV_time_data(U, W, Tv, Ti, Tu, arch=arch_id())
    @info "collecting data for $(SparseMatrixVBC{U, W, Tv, Ti})' * $(Vector{Tu})..."
    @assert arch == arch_id()

    m₀ = ceil(Int, sqrt(first(filter(t->t.type_==:L1Cache, collect(Hwloc.topology_load()))).attr.size)/sizeof(Tv)) #Fill the L1 cache.
    n₀ = m₀
    T = Float64[]
    ms = Int[]
    ns = Int[]
    Ks = Int[]
    Ls = Int[]
    us = Int[]
    ws = Int[]
    for u in 1:U
        for w in 1:W
            ts = Float64[]
            K₀ = cld(m₀, u)
            L₀ = cld(n₀, w)
            for (K, L) in ((K₀, L₀), (2K₀, L₀), (K₀, 2L₀))
                (m, n) = (u * K, w * L)
                A = sparse(ones(Tv, m, n))
                B = SparseMatrixVBC{U, W}(A, pack_stripe(A', EquiChunker(u)), pack_stripe(A, EquiChunker(w)))
                x = ones(Tu, m)
                y = ones(Tu, n)
                mul!(y, B', x)
                t = (@belapsed mul!($y, $B', $x) evals=1_000)
                push!(ms, m)
                push!(ns, n)
                push!(Ks, K)
                push!(Ls, L)
                push!(us, u)
                push!(ws, w)
                push!(T, t)
                @info "u: $u w: $w m: $m n: $n K: $K L: $L t: $t"
            end
        end
    end
    return (ms, ns, Ks, Ls, us, ws, T)
end

@memoize DiskCache(joinpath(@get_scratch!("autotune"), "VBC_TrSpMV_time_params")) function model_SparseMatrixVBC_TrSpMV_time_params(U, W, Tv, Ti, Tu, arch=arch_id())
    @info "calculating cost model for $(SparseMatrixVBC{U, W, Tv, Ti})' * $(Vector{Tu})..."
    (ms, ns, Ks, Ls, us, ws, T) = model_SparseMatrixVBC_TrSpMV_time_data(U, W, Tv, Ti, Tu, arch)
    D = hcat(Ks, ms, Ls, ns, Ks .* Ls, ms .* ns)
    (α_row, β_row, α_col, β_col, α_block, β_block) = qr(Diagonal(1 ./ T) * D, Val(true)) \ ones(length(T))
    @info "results" (α_row, β_row, α_col, β_col, α_block, β_block)
    return (0.0, 0.0, (Line(α_block, 0.0), Line(0.0, 1.0)), (Line(1.0, 0.0), Line(0.0, β_block)))
end