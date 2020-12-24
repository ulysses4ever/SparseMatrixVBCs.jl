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
    for (key, method) in [
        ("strict", StrictChunker(8)),
        ("overlap", OverlapChunker(0.9, 8)),
        ("min blocks", DynamicTotalChunker(model_SparseMatrix1DVBC_blocks(), 8)),
        ("min memory", DynamicTotalChunker(model_SparseMatrix1DVBC_memory(eltype(A), Int), 8)),
        ("min time", DynamicTotalChunker(model_SparseMatrix1DVBC_time(8, eltype(A), Int), 8)),
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

    mdl = BlockComponentCostModel{Int64}((8, 8), 0, 0, (1, identity), (sizeof(Int64), x-> x * sizeof(eltype(A))))
    block_mdl = BlockComponentCostModel{Int64}((8, 8), 0, 0, (1,), (1,))
    for (key, method) in [
        ("1D 2D", AlternatingPacker(DynamicTotalChunker(model_SparseMatrix1DVBC_time(8, eltype(A), Int), 8), EquiChunker(1))),
        ("strict 2D", AlternatingPacker(StrictChunker(8), StrictChunker(8))),
        ("overlap 2D 0.9", AlternatingPacker(OverlapChunker(0.9, 8), OverlapChunker(0.9, 8))),
        ("overlap 2D 0.8", AlternatingPacker(OverlapChunker(0.8, 8), OverlapChunker(0.8, 8))),
        ("overlap 2D 0.7", AlternatingPacker(OverlapChunker(0.7, 8), OverlapChunker(0.7, 8))),
        ("dynamic 2D", AlternatingPacker(DynamicTotalChunker(AffineFillNetCostModel(0, 0, sizeof(Int64), sizeof(eltype(A))), 8), DynamicTotalChunker(mdl, 8))),
        ("blocks 2D", AlternatingPacker(DynamicTotalChunker(AffineFillNetCostModel(0, 0, 0, 1), 8), DynamicTotalChunker(block_mdl, 8))),
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
