using SparseArrays
using ChainPartitioners
using SparseMatrix1DVBCs
using LinearAlgebra
using MatrixDepot
using BenchmarkTools
using Cthulhu
using UnicodePlots
using PrettyTables

for mtx in [
            "Boeing/ct20stif",
            "DIMACS10/chesapeake",
            "Schmid/thermal1",
            "Rothberg/3dtube",
           ]
    A = permutedims(1.0 * sparse(mdopen(mtx).A))

    display(dump(model_SparseMatrix1DVBC_time((1, 4, 8), Float64, Int)))
    println()

    println()
    println()
    println(spy(A, maxwidth=50, maxheight=50, title="$mtx"))
    rows = []
    for (method1, method2) in [(nothing, nothing),
        (SparseMatrix1DVBCs.StrictPartitioner(), ChainPartitioners.StrictChunker(8)),
        (SparseMatrix1DVBCs.OverlapPartitioner(0.9), ChainPartitioners.OverlapChunker(0.9, 8)),
        (SparseMatrix1DVBCs.OptimalPartitioner(SparseMatrix1DVBCs.FixedBlockCost()),
         ChainPartitioners.DynamicTotalChunker(model_SparseMatrix1DVBC_blocks(), 8)),
        (SparseMatrix1DVBCs.OptimalPartitioner(SparseMatrix1DVBCs.BlockRowMemoryCost(Float64, Int)),
         ChainPartitioners.DynamicTotalChunker(model_SparseMatrix1DVBC_memory(Float64, Int), 8)),
        (SparseMatrix1DVBCs.OptimalPartitioner(SparseMatrix1DVBCs.BlockRowTimeCost((1, 4, 8), Float64, Int)),
         ChainPartitioners.DynamicTotalChunker(model_SparseMatrix1DVBC_time((1, 4, 8), Float64, Int), 8)),
    ]
        if (method1, method2) != (nothing, nothing)
            B1 = SparseMatrix1DVBC{(1,4,8)}(A, method1) 
            B2 = SparseMatrix1DVBC{(1,4,8)}(A, method2) 
            @assert size(B1) == size(B2)
            @assert B1.spl == B2.spl
            @assert B1.pos == B2.pos
            @assert B1.ofs == B2.ofs
            @assert B1.val == B2.val
        end
        for method in (method1, method2)
            if method == nothing
                x = rand(size(A, 1))
                y = rand(size(A, 2))
                z = A' * x
                setup_time = 0.0
                mem = sizeof(A.colptr) + sizeof(A.rowval) + sizeof(A.nzval)
                run_time = time(@benchmark TrSpMV!($y, $A, $x))
                
                @assert y == z
                push!(rows, [method setup_time mem run_time])
            else
                B = SparseMatrix1DVBC{(1,4,8)}(A, method)
                setup_time = time(@benchmark SparseMatrix1DVBC{(1,4,8)}($A, $method))

                x = rand(size(A, 1))
                y = rand(size(A, 2))
                z = A' * x

                mem = sizeof(B.spl) + sizeof(B.pos) + sizeof(B.idx) + sizeof(B.ofs) + sizeof(B.val)

                run_time = time(@benchmark TrSpMV!($y, $B, $x))

                @assert y == z
                push!(rows, [method setup_time mem run_time])
            end
        end
    end
    pretty_table(vcat(rows...), ["method", "setuptime", "memory", "runtime"])
end
