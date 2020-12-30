model_SparseMatrix1DVBC_blocks() = AffineNetCostModel(0, 0, 0, 1)

model_SparseMatrix1DVBC_memory(W, Tv, Ti) = ColumnBlockComponentCostModel{Int}(W, 3 * sizeof(Ti), (w) -> sizeof(Ti) + w * sizeof(Tv))

model_SparseMatrix1DVBC_time(W, Tv, Ti) = ColumnBlockComponentCostModel{Float64}(W, model_SparseMatrix1DVBC_time_params(W, Tv, Ti)...)

@memoize @Vault() function model_SparseMatrix1DVBC_time_params(W, Tv, Ti, arch=arch_id())
    @info "calculating $(SparseMatrix1DVBC{W, Tv, Ti}) cost model..."
    @assert arch == arch_id()

    αs = Float64[]
    βs = Float64[]
    ms = [1, 2, 3, 4, 5, 6, 7, 8]
    mem_max = fld(first(filter(t->t.type_==:L1Cache, collect(Hwloc.topology_load()))).attr.size, 2) #Half the L1 cache size. Could be improved.
    mem = model_SparseMatrix1DVBC_memory(W, Tv, Ti)
    for w in 1:W
        ts = Float64[]
        for m in ms
            K = fld(mem_max, mem(w, w*m, m))
            n = w * K
            spl = collect(Ti(1):Ti(w):Ti(n + 1))
            A = SparseMatrix1DVBC{W}(sparse(ones(Tv, m, n)), SplitPartition{Ti}(length(spl) - 1, spl))
            x = ones(m) #TODO add x and y eltype and Δw info to autotuned params
            y = ones(n) #TODO add x and y eltype and Δw info to autotuned params
            mul!(y, A', x)
            t = (@belapsed mul!($y, $A', $x) evals=1_000) / K
            push!(ts, t)
            @info "w: $w m: $m K: $K t: $t"
        end
        m̅ = mean(ms)
        t̅ = mean(ts)
        β = sum((ts .- t̅).*(ms .- m̅)) / sum((ms .- m̅).*(ms .- m̅))
        α = t̅ - β*m̅
        push!(αs, α)
        push!(βs, β)
        @info "w: $w α: $α β: $β"
    end
    for w in W - 1 : -1 : 1
        αs[w] = min(αs[w], αs[w + 1])
        βs[w] = min(βs[w], βs[w + 1])
    end
    @info "αs: $αs"
    @info "βs: $βs"
    @info "done!"
    return ((αs..., ), (βs..., ))
end