using SparseArrays
using SparseMatrixVBCs
using LinearAlgebra
using MatrixDepot
using BenchmarkTools
using Cthulhu
using UnicodePlots
using PrettyTables
using ChainPartitioners
using InteractiveUtils

for mtx in [
            "HB/bcsstk02",
            "Boeing/ct20stif",
            "DIMACS10/chesapeake",
            #"Schmid/thermal1",
            "Rothberg/3dtube",
           ]
    A = permutedims(sparse(mdopen(mtx).A))

    println()
    println()
    println(spy(A, maxwidth=50, maxheight=50, title="$mtx"))
    rows = []

    begin
        x = rand(size(A, 1))
        y = rand(size(A, 2))
        z = A' * x
        setup_time = 0.0
        mem = sizeof(A.colptr) + sizeof(A.rowval) + sizeof(A.nzval)
        run_time = time(@benchmark TrSpMV!($y, $A, $x))
        
        @assert y ≈ z
        push!(rows, ["reference" setup_time mem run_time])
    end

    limit_width(mdl) = ConstrainedCost(mdl, WidthCost(), 8)

    mdl_blocks_1D = model_SparseMatrix1DVBC_blocks()
    mdl_memory_1D = model_SparseMatrix1DVBC_memory(eltype(A), Int)
    mdl_time_1D = model_SparseMatrix1DVBC_TrSpMV_time(8, eltype(A), Int, Float64)

    for (key, method) in [
        ("strict", StrictChunker(8)),
        ("overlap", OverlapChunker(0.9, 8)),
        ("min blocks", DynamicTotalChunker(limit_width(mdl_blocks_1D))),
        ("min memory", DynamicTotalChunker(limit_width(mdl_memory_1D))),
        ("min time", DynamicTotalChunker(limit_width(mdl_time_1D))),
    ]
        B = SparseMatrix1DVBC{8}(A, method)
        setup_time = time(@benchmark SparseMatrix1DVBC{8}($A, $method))

        x = rand(size(A, 1))
        y = rand(size(A, 2))
        z = A' * x


        mem = sizeof(B.Φ) + sizeof(B.pos) + sizeof(B.idx) + sizeof(B.ofs) + sizeof(B.val)

        run_time = time(@benchmark mul!($y, $B', $x, true, false))

        @assert y ≈ z
        push!(rows, [key setup_time mem run_time])
    end

    mdl_blocks_2D = model_SparseMatrixVBC_blocks()
    mdl_memory_2D = model_SparseMatrixVBC_memory(eltype(A), Int)
    mdl_time_2D = model_SparseMatrixVBC_TrSpMV_time(2, 8, 8, eltype(A), Int, Float64)

    for (key, method) in [
        ("1D 2D", AlternatingPacker(DynamicTotalChunker(limit_width(mdl_blocks_1D)), EquiChunker(1))),
        ("strict 2D", AlternatingPacker(StrictChunker(8), StrictChunker(8))),
        ("overlap 2D 0.9", AlternatingPacker(OverlapChunker(0.9, 8), OverlapChunker(0.9, 8))),
        ("overlap 2D 0.8", AlternatingPacker(OverlapChunker(0.8, 8), OverlapChunker(0.8, 8))),
        ("overlap 2D 0.7", AlternatingPacker(OverlapChunker(0.7, 8), OverlapChunker(0.7, 8))),
        ("dynamic blocks 2D", AlternatingPacker(
            DynamicTotalChunker(limit_width(mdl_blocks_1D)),
            DynamicTotalChunker(limit_width(permutedims(mdl_blocks_2D))),
            DynamicTotalChunker(limit_width(mdl_blocks_2D)))),
        ("dynamic memory 2D", AlternatingPacker(
            DynamicTotalChunker(limit_width(mdl_memory_1D)),
            DynamicTotalChunker(limit_width(permutedims(mdl_memory_2D))),
            DynamicTotalChunker(limit_width(mdl_memory_2D)))),
        ("dynamic time 2D", AlternatingPacker(
            DynamicTotalChunker(limit_width(mdl_time_1D)),
            DynamicTotalChunker(limit_width(permutedims(mdl_time_2D))),
            DynamicTotalChunker(limit_width(mdl_time_2D)))),
        ("dynamic memory 2D (1)", AlternatingPacker(
            DynamicTotalChunker(limit_width(mdl_memory_1D)),
            DynamicTotalChunker(limit_width(permutedims(mdl_memory_2D))),
        )),
        ("dynamic memory 2D (2)", AlternatingPacker(
            DynamicTotalChunker(limit_width(mdl_memory_1D)),
            DynamicTotalChunker(limit_width(permutedims(mdl_memory_2D))),
            DynamicTotalChunker(limit_width(mdl_memory_2D)),
            DynamicTotalChunker(limit_width(permutedims(mdl_memory_2D))),
        )),
        ("dynamic memory 2D (4)", AlternatingPacker(
            DynamicTotalChunker(limit_width(mdl_memory_1D)),
            DynamicTotalChunker(limit_width(permutedims(mdl_memory_2D))),
            DynamicTotalChunker(limit_width(mdl_memory_2D)),
            DynamicTotalChunker(limit_width(permutedims(mdl_memory_2D))),
            DynamicTotalChunker(limit_width(mdl_memory_2D)),
            DynamicTotalChunker(limit_width(permutedims(mdl_memory_2D))),
            DynamicTotalChunker(limit_width(mdl_memory_2D)),
            DynamicTotalChunker(limit_width(permutedims(mdl_memory_2D))),
        )),
    ]
        B = SparseMatrixVBC{8, 8}(A, method)
        setup_time = time(@benchmark SparseMatrixVBC{8, 8}($A, $method))

        x = rand(size(A, 1))
        y = rand(size(A, 2))
        z = A' * x

        mem = sizeof(B.Φ) + sizeof(B.Π) + sizeof(B.pos) + sizeof(B.idx) + sizeof(B.ofs) + sizeof(B.val)

        run_time = time(@benchmark mul!($y, $B', $x, true, false))

        @assert y ≈ z
        push!(rows, [key setup_time mem run_time])
    end
    pretty_table(vcat(rows...), ["method", "setuptime", "memory", "runtime"])
end
