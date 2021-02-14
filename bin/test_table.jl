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
            "DIMACS10/chesapeake",
            "HB/bcsstk04",
            "Boeing/ct20stif",
            "Rothberg/3dtube",
            "HB/bcsstk02",
            "Schmid/thermal1",
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
        push!(rows, ["reference" setup_time mem run_time 0])
    end

    w_max = 2

    limit_width(mdl) = ConstrainedCost(mdl, WidthCost(), w_max)

    mdl_blocks_1D = model_SparseMatrix1DVBC_blocks()
    mdl_memory_1D = model_SparseMatrix1DVBC_memory(eltype(A), Int)
    mdl_time_1D = model_SparseMatrix1DVBC_TrSpMV_time(w_max, eltype(A), Int, Float64)

    for (key, method) in [
        ("strict", StrictChunker(w_max)),
        ("overlap", OverlapChunker(0.9, w_max)),
        ("min blocks", DynamicTotalChunker(limit_width(mdl_blocks_1D))),
        ("min memory", DynamicTotalChunker(limit_width(mdl_memory_1D))),
        ("min time", DynamicTotalChunker(limit_width(mdl_time_1D))),
    ]
        B = SparseMatrix1DVBC{w_max}(A, method)
        setup_time = time(@benchmark SparseMatrix1DVBC{$w_max}($A, $method))

        x = rand(size(A, 1))
        y = rand(size(A, 2))
        z = A' * x

        mem = sizeof(B.Φ) + sizeof(B.pos) + sizeof(B.idx) + sizeof(B.ofs) + sizeof(B.val)

        run_time = time(@benchmark mul!($y, $B', $x, true, false))

        run_model = total_value(A, B.Φ, mdl_time_1D)

        @assert y ≈ z
        push!(rows, [key setup_time mem run_time run_model])
    end

    mdl_blocks_2D = model_SparseMatrixVBC_blocks()
    mdl_memory_2D = model_SparseMatrixVBC_memory(eltype(A), Int)
    mdl_time_2D = model_SparseMatrixVBC_TrSpMV_time(w_max, w_max, eltype(A), Int, Float64)

    for (key, method) in [
        ("1D 2D", AlternatingPacker(DynamicTotalChunker(limit_width(mdl_blocks_1D)), EquiChunker(1))),
        ("strict 2D", AlternatingPacker(StrictChunker(w_max), StrictChunker(w_max))),
        ("overlap 2D 0.9", AlternatingPacker(OverlapChunker(0.9, w_max), OverlapChunker(0.9, w_max))),
        ("overlap 2D 0.8", AlternatingPacker(OverlapChunker(0.8, w_max), OverlapChunker(0.8, w_max))),
        ("overlap 2D 0.7", AlternatingPacker(OverlapChunker(0.7, w_max), OverlapChunker(0.7, w_max))),
        ("dynamic blocks 2D", AlternatingPacker(
            DynamicTotalChunker(limit_width(mdl_blocks_1D)),
            DynamicTotalChunker(limit_width(permutedims(mdl_blocks_2D))),
            DynamicTotalChunker(limit_width(mdl_blocks_2D)))),
        ("dynamic memory 2D", AlternatingPacker(
            DynamicTotalChunker(limit_width(mdl_memory_1D)),
            DynamicTotalChunker(limit_width(permutedims(mdl_memory_2D))),
            DynamicTotalChunker(limit_width(mdl_memory_2D)),
        )),
        ("dynamic memory 2D'", AlternatingPacker(
            EquiChunker(1),
            EquiChunker(1),
            DynamicTotalChunker(limit_width(mdl_memory_2D)),
            DynamicTotalChunker(limit_width(permutedims(mdl_memory_2D))),
            DynamicTotalChunker(limit_width(mdl_memory_2D)),
        )),
        ("dynamic time 2D", AlternatingPacker(
            DynamicTotalChunker(limit_width(mdl_time_1D)),
            DynamicTotalChunker(limit_width(permutedims(mdl_time_2D))),
            DynamicTotalChunker(limit_width(mdl_time_2D)),
        )),
        ("dynamic time 2D'", AlternatingPacker(
            EquiChunker(1),
            EquiChunker(1),
            DynamicTotalChunker(limit_width(mdl_time_2D)),
            DynamicTotalChunker(limit_width(permutedims(mdl_time_2D))),
            DynamicTotalChunker(limit_width(mdl_time_2D)),
        )),
    ]
        B = SparseMatrixVBC{w_max, w_max}(A, method)
        setup_time = time(@benchmark SparseMatrixVBC{$w_max, $w_max}($A, $method))

        x = rand(size(A, 1))
        y = rand(size(A, 2))
        z = A' * x

        mem = sizeof(B.Φ) + sizeof(B.Π) + sizeof(B.pos) + sizeof(B.idx) + sizeof(B.ofs) + sizeof(B.val)

        run_time = time(@benchmark mul!($y, $B', $x, true, false))

        run_model = total_value(A, B.Π, B.Φ, mdl_time_2D) + ChainPartitioners.row_component_value(B.Π, mdl_time_2D)

        @assert y ≈ z
        push!(rows, [key setup_time mem run_time run_model])
    end
    pretty_table(vcat(rows...), ["method", "setuptime", "memory", "runtime", "model"])
end
