struct Line{Tv}
    a::Tv
    b::Tv
end

@inline (p::Line)(x) = p.a + p.b * x

model_SparseMatrix1DVBC_blocks() = AffineConnectivityModel(0, 0, 0, 1)

model_SparseMatrix1DVBC_memory(Tv, Ti) = ColumnBlockComponentCostModel{Int}(3 * sizeof(Ti), Line(sizeof(Ti), sizeof(Tv)))

model_SparseMatrix1DVBC_TrSpMV_time(W, Tv, Ti, Tu; kwargs...) = ColumnBlockComponentCostModel{Float64}(model_SparseMatrix1DVBC_TrSpMV_time_params(W, Tv, Ti, Tu; kwargs...)...)

@memoize DiskCache(joinpath(@get_scratch!("autotune"), "1DVBC_TrSpMV_timings")) function model_SparseMatrix1DVBC_TrSpMV_time_data(W, Tv, Ti, Tu; cache=:L2Cache, exceed=false, arch=arch_id())
    @info "collecting data for $(SparseMatrix1DVBC{W, Tv, Ti})' * $(Vector{Tu})..."
    @assert arch == arch_id()

    T = Float64[]
    ms = Int[]
    ns = Int[]
    Ls = Int[]
    ws = Int[]
    qs = Int[]

    _params = nothing
    for w in W:-1:1
        ts = Float64[]

        # Memory Usage (Up To Constants)
        # Ti * L, B.Φ
        # Ti * L, B.pos
        # Ti * L, B.ofs
        # Tu * n, x
        # Tu * m, y
        # Ti * q, B.idx
        # Tv * q * w, B.val
        # Assumptions
        # d = 8
        # n = L * w
        # m = n
        # q = L * d
        # Total Usage
        # ((3 + d) Ti + 2 w Tu + d w Tv)L
        if exceed
            d = 8
            C = first(filter(t->t.type_ == cache, collect(Hwloc.topology_load()))).attr.size * 2
            L₀ = ceil(Int, C/((3 + d) * sizeof(Ti) + (2 * w) * sizeof(Tu) + (d * w) * sizeof(Tv)))
            n₀ = L₀ * w
            m₀ = n₀
            q₀ = L₀ * d
            space = ((m₀, L₀, q₀), (m₀, 2L₀, q₀), (2m₀, L₀, q₀), (m₀, L₀, 2q₀))
        else
            d = 8
            C = first(filter(t->t.type_ == cache, collect(Hwloc.topology_load()))).attr.size / 2
            L₀ = floor(Int, C/((3 + d) * sizeof(Ti) + (2 * w) * sizeof(Tu) + (d * w) * sizeof(Tv)))
            n₀ = L₀ * w
            m₀ = n₀
            q₀ = L₀ * d
            space = ((m₀, L₀, q₀), (m₀, fld(L₀, 2), q₀), (fld(m₀, 2), L₀, q₀), (m₀, L₀, fld(q₀, 2)))
        end
        @assert L₀ >= 4

        for (m, L, q) in space
            n = w * L
            _q = 0
            A_ILh = Set{Tuple{Int, Int}}()
            A_I = Int[]
            A_J = Int[]
            A_V = Tv[]
            while _q < q
                i, l = (rand(1:m), rand(1:L))
                if !((i, l) in A_ILh)
                    push!(A_ILh, (i, l))
                    for j = l * w - w + 1:l * w
                        push!(A_I, i)
                        push!(A_J, j)
                        push!(A_V, rand(Tv))
                    end
                    _q += 1
                end
            end
            A = sparse(A_I, A_J, A_V, m, n)
            B = SparseMatrix1DVBC{W}(A, pack_stripe(A, EquiChunker(w)))
            x = ones(Tu, m)
            y = ones(Tu, n)

            _bench = @benchmarkable mul!($y, $B', $x)
            if _params === nothing
                tune!(_bench)
                _params = params(_bench)
                @info "params" _params
            end
            loadparams!(_bench, _params)
            t = time(run(_bench))

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

@memoize DiskCache(joinpath(@get_scratch!("autotune"), "1DVBC_TrSpMV_time_params")) function model_SparseMatrix1DVBC_TrSpMV_time_params(W, Tv, Ti, Tu; cache=:L2Cache, exceed=false, arch=arch_id())
    @info "calculating cost model for $(SparseMatrix1DVBC{W, Tv, Ti})' * $(Vector{Tu})..."
    (ms, ns, Ls, ws, qs, T) = model_SparseMatrix1DVBC_TrSpMV_time_data(W, Tv, Ti, Tu; cache = cache, exceed = exceed, arch = arch)
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

    @info "results" α_row α_col β_col
    @info "done!"
    return (α_col, β_col)
end

model_SparseMatrixVBC_blocks() = BlockComponentCostModel{Int}(0, 0, (1,), (1, ))

model_SparseMatrixVBC_memory(Tv, Ti) = BlockComponentCostModel{Int}(sizeof(Ti), 3 * sizeof(Ti), (Line(1, 0), Line(0, 1)), (Line(sizeof(Ti), 0), Line(0, sizeof(Tv))))

model_SparseMatrixVBC_TrSpMV_time(R, U, W, Tv, Ti, Tu; kwargs...) = BlockComponentCostModel{Float64}(model_SparseMatrixVBC_TrSpMV_time_params(R, U, W, Tv, Ti, Tu; kwargs...)...)

@memoize DiskCache(joinpath(@get_scratch!("autotune"), "VBC_TrSpMV_timings")) function model_SparseMatrixVBC_TrSpMV_time_data(U, W, Tv, Ti, Tu; cache=:L2Cache, exceed=false, arch=arch_id())
    @info "collecting data for $(SparseMatrixVBC{U, W, Tv, Ti})' * $(Vector{Tu})..."
    @assert arch == arch_id()

    T = Float64[]
    ms = Int[]
    ns = Int[]
    Ks = Int[]
    Ls = Int[]
    us = Int[]
    ws = Int[]
    qs = Int[]

    _params = nothing
    for u in U:-1:1
        for w in W:-1:1
            # Memory Usage (Up To Constants)
            # Ti * K, B.Π
            # Ti * L, B.Φ
            # Ti * L, B.pos
            # Ti * L, B.ofs
            # Tu * n, x
            # Tu * m, y
            # Ti * q, B.idx
            # Tv * q * u * w, B.val
            # Assumptions
            # d = 8
            # m = K * u
            # n = L * w
            # m = n
            # q = L * d
            # Total Usage
            # ((3 + (w/u) + d) Ti + 2 w Tu + d u w Tv)L < C/2
            if exceed
                d = 8
                C = first(filter(t->t.type_ == cache, collect(Hwloc.topology_load()))).attr.size * 2
                L₀ = ceil(Int, C/((3 + d + (w/u)) * sizeof(Ti) + (2 * w) * sizeof(Tu) + (d * u * w) * sizeof(Tv)))
                n₀ = L₀ * w
                K₀ = cld(n₀, u)
                m₀ = K₀ * u
                q₀ = L₀ * d
                space = ((K₀, L₀, q₀), (K₀, 2L₀, q₀), (2K₀, L₀, q₀), (K₀, L₀, 2q₀))
            else
                d = 8
                C = first(filter(t->t.type_ == cache, collect(Hwloc.topology_load()))).attr.size / 2
                L₀ = floor(Int, C/((3 + d + (w/u)) * sizeof(Ti) + (2 * w) * sizeof(Tu) + (d * u * w) * sizeof(Tv)))
                n₀ = L₀ * w
                K₀ = fld(n₀, u)
                m₀ = K₀ * u
                q₀ = L₀ * d
                space = ((K₀, L₀, q₀), (K₀, fld(L₀, 2), q₀), (fld(K₀, 2), L₀, q₀), (K₀, L₀, fld(q₀, 2)))
            end

            @assert L₀ >= 4
            @assert K₀ >= 4

            for (K, L, q) in space
                (m, n) = (u * K, w * L)
                _q = 0
                A_KLh = Set{Tuple{Int, Int}}()
                A_I = Int[]
                A_J = Int[]
                A_V = Tv[]
                while _q < q
                    k, l = (rand(1:K), rand(1:L))
                    if !((k, l) in A_KLh)
                        push!(A_KLh, (k, l))
                        for i = k * u - u + 1: k * u, j = l * w - w + 1:l * w
                            push!(A_I, i)
                            push!(A_J, j)
                            push!(A_V, rand(Tv))
                        end
                        _q += 1
                    end
                end
                A = sparse(A_I, A_J, A_V, m, n)
                B = SparseMatrixVBC{U, W}(A, pack_stripe(A', EquiChunker(u)), pack_stripe(A, EquiChunker(w)))
                x = ones(Tu, m)
                y = ones(Tu, n)

                _bench = @benchmarkable mul!($y, $B', $x)
                if _params === nothing
                    tune!(_bench)
                    _params = params(_bench)
                    @info "params" _params
                end
                loadparams!(_bench, _params)
                t = time(run(_bench))

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

@memoize DiskCache(joinpath(@get_scratch!("autotune"), "VBC_TrSpMV_time_params")) function model_SparseMatrixVBC_TrSpMV_time_params(R, U, W, Tv, Ti, Tu; cache=:L2Cache, exceed=false, arch=arch_id())
    @info "calculating cost model for $(SparseMatrixVBC{U, W, Tv, Ti})' * $(Vector{Tu})..."
    (ms, ns, Ks, Ls, us, ws, qs, T) = model_SparseMatrixVBC_TrSpMV_time_data(U, W, Tv, Ti, Tu; cache = cache, exceed = exceed, arch = arch)
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
