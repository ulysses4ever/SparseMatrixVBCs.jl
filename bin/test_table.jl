using SparseArrays
using SparseMatrix1DVBCs
using LinearAlgebra
using MatrixDepot
using BenchmarkTools
using Cthulhu
using UnicodePlots
using PrettyTables

for mtx in [
            "DIMACS10/chesapeake",
            #"Schmid/thermal1",
            "Boeing/ct20stif",
            "Rothberg/3dtube",
           ]
    A = permutedims(1.0 * sparse(mdopen(mtx).A))

    println()
    println()
    println(spy(A, maxwidth=50, maxheight=50, title="$mtx"))
    rows = []
    for method in [nothing,
                   SparseMatrix1DVBCs.EminentPartitioner(),
                   SparseMatrix1DVBCs.OverlapPartitioner(0.9),
                   SparseMatrix1DVBCs.OptimalPartitioner(SparseMatrix1DVBCs.BlockRowMemoryCost(Float64, Int)),
                  ]
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
    pretty_table(vcat(rows...), ["method", "setuptime", "memory", "runtime"])
end