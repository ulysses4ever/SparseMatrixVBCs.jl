using SparseArrays
using SparseMatrix1DVBCs
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

    mdl_blocks = model_SparseMatrix1DVBC_blocks(8)
    mdl_memory = model_SparseMatrix1DVBC_memory(8, eltype(A), Int)
    mdl_time = model_SparseMatrix1DVBC_time(8, eltype(A), Int)

    for (key, method) in [
        ("strict", StrictChunker(8)),
        ("overlap", OverlapChunker(0.9, 8)),
        ("min blocks", DynamicTotalChunker(mdl_blocks, 8)),
        ("min memory", DynamicTotalChunker(mdl_memory, 8)),
        ("min time", DynamicTotalChunker(mdl_time, 8)),
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

    mdl_blocks_2D = model_SparseMatrixVBC_blocks(8, 8)
    mdl_memory_2D = model_SparseMatrixVBC_memory(8, 8, eltype(A), Int)
    mdl_time_2D = model_SparseMatrixVBC_time(2, 8, 8, eltype(A), Int)
    for (key, method) in [
        ("1D 2D", AlternatingPacker(DynamicTotalChunker(model_SparseMatrix1DVBC_time(8, eltype(A), Int), 8), EquiChunker(1))),
        ("strict 2D", AlternatingPacker(StrictChunker(8), StrictChunker(8))),
        ("overlap 2D 0.9", AlternatingPacker(OverlapChunker(0.9, 8), OverlapChunker(0.9, 8))),
        ("overlap 2D 0.8", AlternatingPacker(OverlapChunker(0.8, 8), OverlapChunker(0.8, 8))),
        ("overlap 2D 0.7", AlternatingPacker(OverlapChunker(0.7, 8), OverlapChunker(0.7, 8))),
        ("dynamic blocks 2D", AlternatingPacker(DynamicTotalChunker(mdl_blocks, 8), DynamicTotalChunker(permutedims(mdl_blocks_2D), 8), DynamicTotalChunker(mdl_blocks_2D, 8))),
        ("dynamic memory 2D", AlternatingPacker(DynamicTotalChunker(mdl_memory, 8), DynamicTotalChunker(permutedims(mdl_memory_2D), 8), DynamicTotalChunker(mdl_memory_2D, 8))),
        ("dynamic time 2D", AlternatingPacker(DynamicTotalChunker(mdl_time, 8), DynamicTotalChunker(permutedims(mdl_time_2D), 8), DynamicTotalChunker(mdl_time_2D, 8))),
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
