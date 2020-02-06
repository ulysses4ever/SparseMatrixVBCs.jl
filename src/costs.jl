function partitioncost(g::G, A_prt::Partition) where {G}
    @inbounds begin
        A_spl = A_prt.spl
        A_pos = A_prt.pos
        k = length(s)

        c = zero(g)
        for jj = 1:k
            c += g(A_spl[jj + 1] - A_spl[jj], A_pos[jj + 1] - A_pos[jj])
        end
        return c
    end
end



struct FixedBlockCost end

Base.zero(::FixedBlockCost) = 0

@inline (::FixedBlockCost)(w, d) = d



struct BlockRowMemoryCost{Tv, Ti} end

BlockRowMemoryCost(Tv, Ti) = BlockRowMemoryCost{Tv, Ti}()

@inline (::BlockRowMemoryCost{Tv, Ti})(w, d) where {Tv, Ti} = 3 * sizeof(Ti) + d * (sizeof(Ti) + w * sizeof(Tv))

Base.zero(g::BlockRowMemoryCost) = g(false, false)

function Base.show(io::IO, g::BlockRowMemoryCost{Tv, Ti}) where {Tv, Ti}
    print(io, "BlockRowMemoryCost")
    print(io, (Tv, Ti))
end



struct BlockRowTimeCost{Ws, Tv, Ti}
    αs::Vector{Float64}
    βs::Vector{Float64}
    function BlockRowTimeCost{Ws, Tv, Ti}() where {Ws, Tv, Ti}
        cache = Dict()
        if isfile(cachefile)
            cache = BSON.load(cachefile)
        end
        if !(BlockRowTimeCost{Ws, Tv, Ti} in keys(cache))
            @info "calculating $(BlockRowTimeCost{Ws, Tv, Ti}) model..."

            αs = Float64[]
            βs = Float64[]
            ms = [1, 2, 3, 4, 5, 6, 7, 8]
            mem_max = fld(first(filter(t->t.type_==:L1Cache, collect(Hwloc.topology_load()))).attr.size, 2) #Half the L1 cache size. Could be improved.
            mem = BlockRowMemoryCost{Tv, Ti}()
            for w in 1:max(Ws...)
                ts = Float64[]
                for m in ms
                    k = fld(mem_max, mem(w, max(m, 1)))
                    n = w * k
                    spl = collect(Ti(1):Ti(w):Ti(n + 1))
                    ofs = 1 .+ ((spl .- 1) .* m)
                    pos = 1 .+ (fld.((ofs .- 1), w))
                    A = SparseMatrix1DVBC{Ws}(sparse(ones(Tv, m, n)), Partition{Ti}(spl, pos, ofs))
                    x = ones(Tv, m)
                    y = ones(Tv, n)
                    TrSpMV!(y, A, x)
                    t = (@belapsed TrSpMV!($y, $A, $x) evals=1_000) / k
                    push!(ts, t)
                    @info "w: $w m: $m k: $k t: $t"
                end
                m̅ = mean(ms)
                t̅ = mean(ts)
                β = sum((ts .- t̅).*(ms .- m̅)) / sum((ms .- m̅).*(ms .- m̅))
                α = t̅ - β*m̅
                push!(αs, α)
                push!(βs, β)
                @info "w: $w α: $α β: $β"
            end
            for w in max(Ws...) -1:-1:1
                αs[w] = min(αs[w], αs[w + 1])
                βs[w] = min(βs[w], βs[w + 1])
            end
            @info "αs: $αs"
            @info "βs: $βs"
            @info "done!"
            cache[BlockRowTimeCost{Ws, Tv, Ti}] = new{Ws, Tv, Ti}(αs, βs)
        end
        BSON.bson(cachefile, cache)
        return cache[BlockRowTimeCost{Ws, Tv, Ti}]
    end
end

BlockRowTimeCost(Ws, Tv, Ti) = BlockRowTimeCost{Ws, Tv, Ti}()

#Base.@propagate_inbounds (g::BlockRowTimeCost)(w, d)::Float64 = g.αs[w] + g.βs[w]*d
@inline (g::BlockRowTimeCost)(w, d)::Float64 = g.αs[w] + g.βs[w]*d

Base.zero(g::BlockRowTimeCost) = zero(Float64)

function Base.show(io::IO, ::BlockRowTimeCost{Ws, Tv, Ti}) where {Ws, Tv, Ti}
    print(io, "BlockRowTimeCost")
    print(io, (Ws, Tv, Ti))
end