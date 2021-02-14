struct Line{Tv}
    a::Tv
    b::Tv
end

@inline (p::Line)(x) = p.a + p.b * x

model_SparseMatrix1DVBC_blocks() = AffineNetCostModel(0, 0, 0, 1)

model_SparseMatrix1DVBC_memory(Tv, Ti) = ColumnBlockComponentCostModel{Int}(3 * sizeof(Ti), Line(sizeof(Ti), sizeof(Tv)))

model_SparseMatrix1DVBC_TrSpMV_time(W, Tv, Ti, Tu) = ColumnBlockComponentCostModel{Float64}(model_SparseMatrix1DVBC_TrSpMV_time_params(W, Tv, Ti, Tu)...)

model_SparseMatrix1DVBC_TrSpMV_fancy(W, Tv, Ti, Tu) = ColumnBlockComponentCostModel{Float64}(model_SparseMatrix1DVBC_TrSpMV_fancy_params(W, Tv, Ti, Tu)...)

@memoize DiskCache(joinpath(@get_scratch!("autotune"), "1DVBC_TrSpMV_timings")) function model_SparseMatrix1DVBC_TrSpMV_time_data(W, Tv, Ti, Tu, arch=arch_id())
    @info "collecting data for $(SparseMatrix1DVBC{W, Tv, Ti})' * $(Vector{Tu})..."
    @assert arch == arch_id()

    m₀ = ceil(Int, first(filter(t->t.type_==:L2Cache, collect(Hwloc.topology_load()))).attr.size/min(sizeof(Ti), sizeof(Tu)))
    b₀ = ceil(Int, first(filter(t->t.type_==:L2Cache, collect(Hwloc.topology_load()))).attr.size/sizeof(Tv))
    b₀ = max(b₀, 2m₀)
    n₀ = m₀
    T = Float64[]
    ms = Int[]
    ns = Int[]
    Ls = Int[]
    ws = Int[]
    qs = Int[]

    for w in 1:W
        ts = Float64[]
        L₀ = cld(n₀, w)
        for (m, L, b) in ((m₀, L₀, b₀), (m₀, 2L₀, b₀), (2m₀, L₀, b₀), (m₀, L₀, 2b₀))
            n = w * L
            A = sparse(zeros(Tv, m, n))
            q = 0
            while q < cld(b, w)
                i, l = (rand(1:m), rand(1:L))
                if A[i, l * w] == zero(Tv)
                    A[i, l * w - w + 1:l * w] .= one(Tv)
                    q += 1
                end
            end
            B = SparseMatrix1DVBC{W}(A, pack_stripe(A, EquiChunker(w)))
            x = ones(Tu, m)
            y = ones(Tu, n)
            mul!(y, B', x)
            t = (@belapsed mul!($y, $B', $x) evals=1)
            push!(ms, m)
            push!(ns, n)
            push!(Ls, L)
            push!(ws, w)
            push!(qs, q)
            push!(T, t)
            @info "w: $w m: $m n: $n L: $L q: $q t: $t"
        end
    end
    @info "done!"
    return (ms, ns, Ls, ws, qs, T)
end

@memoize DiskCache(joinpath(@get_scratch!("autotune"), "1DVBC_TrSpMV_time_params")) function model_SparseMatrix1DVBC_TrSpMV_time_params(W, Tv, Ti, Tu, arch=arch_id())
    @info "calculating cost model for $(SparseMatrix1DVBC{W, Tv, Ti})' * $(Vector{Tu})..."
    (ms, ns, Ls, ws, qs, T) = model_SparseMatrix1DVBC_TrSpMV_time_data(W, Tv, Ti, Tu, arch)
    D = hcat(ms, Ls, ns, qs, qs .* ws)
    (α_row, α_col₀, α_col₁, β_col₀, β_col₁) = qr(Diagonal(1 ./ T) * D, Val(true)) \ ones(length(T))
    @info "results" α_row α_col₀ α_col₁ β_col₀ β_col₁
    @info "done!"
    return (0.0, Line(β_col₀, β_col₁))
end

@memoize DiskCache(joinpath(@get_scratch!("autotune"), "1DVBC_TrSpMV_fancy_params")) function model_SparseMatrix1DVBC_TrSpMV_fancy_params(W, Tv, Ti, Tu, arch=arch_id())
    @info "calculating cost model for $(SparseMatrix1DVBC{W, Tv, Ti})' * $(Vector{Tu})..."
    (ms, ns, Ls, ws, qs, T) = model_SparseMatrix1DVBC_TrSpMV_time_data(W, Tv, Ti, Tu, arch)
    d = Vector{Float64}[]
    for i = 1:length(T) #Create one-hot representation
        d_row = ms[i]
        d_col = zeros(W)
        d_col[ws[i]] = Ls[i]
        d_block = zeros(W)
        d_block[ws[i]] = qs[i]
        push!(d, [d_row; d_col; d_block])
    end
    D = permutedims(reduce(hcat, d))
    P = qr(Diagonal(1 ./ T) * D, Val(true)) \ ones(length(T))
    α_row = P[1]
    α_col = P[2:1 + W]
    β_col = P[2 + W:end]

    #monotonize
    for w = 2:W
        α_col[w] = max(α_col[w], α_col[w - 1])
        β_col[w] = max(β_col[w], β_col[w - 1])
    end

    #tiebreak
    α_col .+= [minimum(α_col) * 0.0001 * w for w = 1:W]
    β_col .+= [minimum(β_col) * 0.0001 * w for w = 1:W]

    @info "results" α_row α_col β_col
    @info "done!"
    return (α_col, β_col)
end

model_SparseMatrixVBC_blocks() = BlockComponentCostModel{Int}(0, 0, (1,), (1, ))

model_SparseMatrixVBC_memory(Tv, Ti) = BlockComponentCostModel{Int}(sizeof(Ti), 3 * sizeof(Ti), (Line(1, 0), Line(0, 1)), (Line(sizeof(Ti), 0), Line(0, sizeof(Tv))))

model_SparseMatrixVBC_TrSpMV_time(U, W, Tv, Ti, Tu) = BlockComponentCostModel{Float64}(model_SparseMatrixVBC_TrSpMV_time_params(U, W, Tv, Ti, Tu)...)

model_SparseMatrixVBC_TrSpMV_fancy(R, U, W, Tv, Ti, Tu) = BlockComponentCostModel{Float64}(model_SparseMatrixVBC_TrSpMV_fancy_params(R, U, W, Tv, Ti, Tu)...)

@memoize DiskCache(joinpath(@get_scratch!("autotune"), "VBC_TrSpMV_timings")) function model_SparseMatrixVBC_TrSpMV_time_data(U, W, Tv, Ti, Tu, arch=arch_id())
    @info "collecting data for $(SparseMatrixVBC{U, W, Tv, Ti})' * $(Vector{Tu})..."
    @assert arch == arch_id()

    m₀ = ceil(Int, first(filter(t->t.type_==:L2Cache, collect(Hwloc.topology_load()))).attr.size/min(sizeof(Ti), sizeof(Tu))) #Fill the L2 cache.
    b₀ = ceil(Int, 2 * first(filter(t->t.type_==:L2Cache, collect(Hwloc.topology_load()))).attr.size/sizeof(Tv))
    b₀ = max(b₀, 2m₀)
    n₀ = m₀
    T = Float64[]
    ms = Int[]
    ns = Int[]
    Ks = Int[]
    Ls = Int[]
    us = Int[]
    ws = Int[]
    qs = Int[]
    for u in 1:U
        for w in 1:W
            ts = Float64[]
            K₀ = cld(m₀, u)
            L₀ = cld(n₀, w)
            for (K, L, b) in ((K₀, L₀, b₀), (K₀, 2L₀, b₀), (2K₀, L₀, b₀), (K₀, L₀, 2b₀))
                (m, n) = (u * K, w * L)
                A = sparse(zeros(Tv, m, n))
                q = 0
                while q < cld(b, u * w)
                    k, l = (rand(1:K), rand(1:L))
                    if A[k * u, l * w] == zero(Tv)
                        A[k * u - u + 1: k * u, l * w - w + 1:l * w] .= one(Tv)
                        q += 1
                    end
                end
                B = SparseMatrixVBC{U, W}(A, pack_stripe(A', EquiChunker(u)), pack_stripe(A, EquiChunker(w)))
                x = ones(Tu, m)
                y = ones(Tu, n)
                mul!(y, B', x)
                t = (@belapsed mul!($y, $B', $x) evals=1)
                push!(ms, m)
                push!(ns, n)
                push!(Ks, K)
                push!(Ls, L)
                push!(us, u)
                push!(ws, w)
                push!(qs, q)
                push!(T, t)
                @info "u: $u w: $w m: $m n: $n K: $K L: $L q: $q t: $t"
            end
        end
    end
    @info "done!"
    return (ms, ns, Ks, Ls, us, ws, qs, T)
end

@memoize DiskCache(joinpath(@get_scratch!("autotune"), "VBC_TrSpMV_time_params")) function model_SparseMatrixVBC_TrSpMV_time_params(U, W, Tv, Ti, Tu, arch=arch_id())
    @info "calculating cost model for $(SparseMatrixVBC{U, W, Tv, Ti})' * $(Vector{Tu})..."
    (ms, ns, Ks, Ls, us, ws, qs, T) = model_SparseMatrixVBC_TrSpMV_time_data(U, W, Tv, Ti, Tu, arch)
    D = hcat(Ks, ms, Ls, ns, qs, qs .* us .* ws)
    (α_row₀, α_row₁, α_col₀, α_col₁, β_col₀, β_col₁) = qr(Diagonal(1 ./ T) * D, Val(true)) \ ones(length(T))
    @info "results" α_row₀ α_row₁ α_col₀ α_col₁ β_col₀ β_col₁
    @info "done!"
    return (0.0, 0.0, (Line(β_col₀, 0.0), Line(0.0, 1.0)), (Line(1.0, 0.0), Line(0.0, β_col₁)))
end

@memoize DiskCache(joinpath(@get_scratch!("autotune"), "VBC_TrSpMV_fancy_params")) function model_SparseMatrixVBC_TrSpMV_fancy_params(R, U, W, Tv, Ti, Tu, arch=arch_id())
    @info "calculating cost model for $(SparseMatrixVBC{U, W, Tv, Ti})' * $(Vector{Tu})..."
    (ms, ns, Ks, Ls, us, ws, qs, T) = model_SparseMatrixVBC_TrSpMV_time_data(U, W, Tv, Ti, Tu, arch)
    d = Vector{Float64}[]
    for i = 1:length(T) #Create one-hot representation
        d_row = zeros(U)
        d_row[us[i]] = Ks[i]
        d_col = zeros(W)
        d_col[ws[i]] = Ls[i]
        d_block = zeros(U, W)
        d_block[us[i], ws[i]] = qs[i]
        push!(d, [d_row; d_col; reshape(d_block, :)])
    end
    D = permutedims(reduce(hcat, d))
    P = qr(Diagonal(1 ./ T) * D, Val(true)) \ ones(length(T))
    α_row = [P[1:U]...,]
    α_col = [P[U + 1:U + W]...,]
    β = collect(reshape(P[U + W + 1:end], U, W))

    #monotonize
    for w = 2:W
        β[1, w] = max(β[1, w], β[1, w - 1])
    end
    for u = 2:U
        β[u, 1] = max(β[u, 1], β[u - 1, 1])
        for w = 2:W
            β[u, w] = max(β[u, w], β[u - 1, w - 1], β[u, w - 1])
        end
    end

    #tiebreak
    β .+= [minimum(β) * 0.0001 * u * w for u = 1:U, w = 1:W]

    F = svd(β)
    β_row = ((F.U[:,r] for r = 1:R)...,)
    β_col = ((F.S[r] * F.V[:,r] for r = 1:R)...,)

    β_reconstruct = [sum(β_row[r][u] * β_col[r][w] for r = 1:R) for u = 1:U, w = 1:W]
    β_error = maximum((β_reconstruct .- β)./β)
    @info "raw" β_error β β_reconstruct

    @info "results" α_row α_col β_row β_col
    @info "done!"
    return (α_row, α_col, β_row, β_col)
end