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



#=
struct VBCCSCTimeCost{Ws, Tv, Ti}
    gs::Vector{Interpolations.Extrapolation{Float64,1,Interpolations.GriddedInterpolation{Float64,1,Float64,Gridded{Linear},Tuple{Array{Int64,1}}},Gridded{Linear},Line{Nothing}}}
    function VBCCSCTimeCost{Ws, Tv, Ti}() where {Ws, Tv, Ti}
        cache = Dict()
        if isfile(cachefile)
            cache = BSON.load(cachefile)
        end
        if !(VBCCSCTimeCost{Ws, Tv, Ti} in keys(cache))
            @info "calculating $(SparseMatrix1DVBC{Ws, Tv, Ti}) runtime cost model..."
            fs = Interpolations.Extrapolation{Float64,1,Interpolations.GriddedInterpolation{Float64,1,Float64,Gridded{Linear},Tuple{Array{Int64,1}}},Gridded{Linear},Line{Nothing}}[]
            ms = [0, 1, 2, 4, 8, 32, 128]
            q = cld(first(filter(t->t.type_==:L3Cache, collect(Hwloc.topology_load()))).attr.size, sizeof(Tv)) #Attempt to fill the L3 cache. Could be improved.
            for w in 1:max(Ws...)
                ts = Float64[]
                for m in ms
                    k = cld(q, w * max(m, 1)) #Gotta hit that l3 cache yo!
                    n = w * k
                    spl = collect(Ti(1):Ti(w):Ti(n + 1))
                    ofs = 1 .+ ((spl .- 1) .* m)
                    pos = 1 .+ (fld.((ofs .- 1), w))
                    A = SparseMatrix1DVBC{Ws}(sparse(ones(Tv, m, n)), Blocks{Ti}(spl, pos, ofs))
                    x = ones(Tv, m)
                    y = ones(Tv, n)
                    t = time(@benchmark TrSpMV!($y, $A, $x) seconds=10) / k
                    push!(ts, t)
                    @info "w: $w m: $m t: $t"
                end
                push!(fs, LinearInterpolation(ms, ts, extrapolation_bc = Natural()))
            end
            @info "done!"
            cache[VBCCSCTimeCost{Ws, Tv, Ti}] = new{Ws, Tv, Ti}(fs)
        end
        BSON.bson(cachefile, cache)
        return cache[VBCCSCTimeCost{Ws, Tv, Ti}]
    end
end

VBCCSCTimeCost(Ws, Tv, Ti) = VBCCSCTimeCost{Ws, Tv, Ti}()

@inline (f::VBCCSCTimeCost)(w, d)::Float64 = f.fs[w](d)

Base.zero(f::VBCCSCTimeCost) = zero(Float64)

function Base.show(io::IO, f::VBCCSCTimeCost{Ws, Tv, Ti}) where {Ws, Tv, Ti}
    print(io, "VBCCSCTimeCost")
    print(io, (Ws, Tv, Ti))
end
=#