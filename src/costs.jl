model_SparseMatrix1DVBC_blocks() = AffineFillNetCostModel(0, 0, 0, 1)

model_SparseMatrix1DVBC_memory(Tv, Ti) = AffineFillNetCostModel(3 * sizeof(Ti), 0, sizeof(Tv), sizeof(Ti))

struct SparseMatrix1DVBCTimeModel{Ws, Tv, Ti} <: AbstractNetCostModel
    αs::Vector{Float64}
    βs::Vector{Float64}
    function SparseMatrix1DVBCTimeModel{Ws, Tv, Ti}() where {Ws, Tv, Ti}
        cache = Dict()
        if isfile(cachefile)
            cache = BSON.load(cachefile)
        end
        if !(SparseMatrix1DVBCTimeModel{Ws, Tv, Ti} in keys(cache))
            @info "calculating $(SparseMatrix1DVBCTimeModel{Ws, Tv, Ti}) model..."

            αs = Float64[]
            βs = Float64[]
            ms = [1, 2, 3, 4, 5, 6, 7, 8]
            mem_max = fld(first(filter(t->t.type_==:L1Cache, collect(Hwloc.topology_load()))).attr.size, 2) #Half the L1 cache size. Could be improved.
            mem = model_SparseMatrix1DVBC_memory(Tv, Ti)
            for w in 1:max(Ws...)
                ts = Float64[]
                for m in ms
                    K = fld(mem_max, mem(w, w*m, m))
                    n = w * K
                    spl = collect(Ti(1):Ti(w):Ti(n + 1))
                    A = SparseMatrix1DVBC{Ws}(sparse(ones(Tv, m, n)), SplitPartition{Ti}(length(spl) - 1, spl))
                    x = ones(Tv, m)
                    y = ones(Tv, n)
                    TrSpMV!(y, A, x)
                    t = (@belapsed TrSpMV!($y, $A, $x) evals=1_000) / K
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
            for w in max(Ws...) - 1 : -1 : 1
                αs[w] = min(αs[w], αs[w + 1])
                βs[w] = min(βs[w], βs[w + 1])
            end
            @info "αs: $αs"
            @info "βs: $βs"
            @info "done!"
            cache[SparseMatrix1DVBCTimeModel{Ws, Tv, Ti}] = new{Ws, Tv, Ti}(αs, βs)
            BSON.bson(cachefile, cache)
        end
        return cache[SparseMatrix1DVBCTimeModel{Ws, Tv, Ti}]
    end
end

Base.@propagate_inbounds (mdl::SparseMatrix1DVBCTimeModel)(x_width, x_work, x_net)::Float64 = mdl.αs[x_width] + mdl.βs[x_width]*x_net

ChainPartitioners.cost_type(::Type{SparseMatrix1DVBCTimeModel{Ws, Tv, Ti}}) where {Ws, Tv, Ti} = Float64

model_SparseMatrix1DVBC_time(Ws, Tv, Ti) = SparseMatrix1DVBCTimeModel{Ws, Tv, Ti}()

function Base.show(io::IO, ::SparseMatrix1DVBCTimeModel{Ws, Tv, Ti}) where {Ws, Tv, Ti}
    print(io, "SparseMatrix1DVBCTimeModel")
    print(io, (Ws, Tv, Ti))
end