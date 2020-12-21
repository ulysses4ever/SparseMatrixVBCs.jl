using Test
using Random
using ChainPartitioners
using SparseArrays
using SparseMatrix1DVBCs

include("matrices.jl")

@testset "SparseMatrix1DVBCs" begin
    Random.seed!(0xDEADBEEF)
    for A in [
        collect(values(matrices));
        reshape([sprand(m, n, 0.2) for m = [1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17], n = [1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17], trial = 1:4], :);
    ]
        A = SparseMatrixCSC(A * 1.0) #TODO generic eltypes and symmetric constructors
        for method in [
            StrictChunker(4),
            OverlapChunker(0.9, 4),
            DynamicTotalChunker(model_SparseMatrix1DVBC_blocks(), 4),
            DynamicTotalChunker(model_SparseMatrix1DVBC_memory(Float64, Int), 4),
            DynamicTotalChunker(model_SparseMatrix1DVBC_time((1, 2, 4), Float64, Int), 4),
        ]
            (m, n) = size(A)
            B = SparseMatrix1DVBC{(1, 2, 4)}(A, method)
            x = zeros(eltype(A), m)
            y_ref = Vector{eltype(A)}(undef, n)
            y_test = Vector{eltype(A)}(undef, n)
            for i = 1:m
                x[i] = true
                fill!(y_ref, false)
                fill!(y_test, false)
                TrSpMV!(y_ref, A, x) 
                TrSpMV!(y_test, B, x) 
                @test y_ref == y_test
                x[i] = false
            end
        end
    end
end
