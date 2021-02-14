using Test
using Random
using ChainPartitioners
using SparseArrays
using SparseMatrixVBCs
using LinearAlgebra

include("matrices.jl")

@testset "SparseMatrixVBCs" begin
    Random.seed!(0xDEADBEEF)
    for A in [
        collect(values(matrices));
        reshape([sprand(m, n, 0.2) for m = [1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17], n = [1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17], trial = 1:4], :);
        reshape([sprand(Bool, m, n, 0.2) for m = [1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17], n = [1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17], trial = 1:4], :);
        reshape([sprand(Int32, m, n, 0.2) for m = [1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17], n = [1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17], trial = 1:4], :);
    ]
        A = SparseMatrixCSC(A) #TODO symmetric constructors
        for method in [
            StrictChunker(4),
            OverlapChunker(0.9, 4),
            DynamicTotalChunker(ConstrainedCost(model_SparseMatrix1DVBC_blocks(), WidthCost(), 4)),
            DynamicTotalChunker(ConstrainedCost(model_SparseMatrix1DVBC_memory(Float64, Int), WidthCost(), 4)),
            #DynamicTotalChunker(model_SparseMatrix1DVBC_time(4, Float64, Int), 4),
        ]
            (m, n) = size(A)
            B = SparseMatrix1DVBC{4}(A, method)

            x = zeros(eltype(A), n)
            y_ref = A * x
            y_test = A * x
            for j = 1:n
                x[j] = true
                fill!(y_ref, false)
                fill!(y_test, false)
                LinearAlgebra.mul!(y_ref, A, x, true, false) 
                LinearAlgebra.mul!(y_test, B, x, true, false) 
                @test y_ref == y_test
                x[j] = false
            end

            x = zeros(eltype(A), m)
            y_ref = A' * x
            y_test = A' * x
            for i = 1:m
                x[i] = true
                fill!(y_ref, false)
                fill!(y_test, false)
                LinearAlgebra.mul!(y_ref, A', x, true, false) 
                LinearAlgebra.mul!(y_test, B', x, true, false) 
                @test y_ref == y_test
                x[i] = false
            end
        end

        for method in [
            AlternatingPacker(StrictChunker(4), StrictChunker(4)),
            AlternatingPacker(OverlapChunker(0.9, 4), OverlapChunker(0.9, 4)),
        ]
            (m, n) = size(A)
            B = SparseMatrixVBC{4, 4}(A, method)

            x = zeros(eltype(A), n)
            y_ref = A * x
            y_test = A * x
            for j = 1:n
                x[j] = true
                fill!(y_ref, false)
                fill!(y_test, false)
                LinearAlgebra.mul!(y_ref, A, x, true, false) 
                LinearAlgebra.mul!(y_test, B, x, true, false) 
                @test y_ref == y_test
                x[j] = false
            end

            x = zeros(eltype(A), m)
            y_ref = A' * x
            y_test = A' * x
            for i = 1:m
                x[i] = true
                fill!(y_ref, false)
                fill!(y_test, false)
                LinearAlgebra.mul!(y_ref, A', x, true, false) 
                LinearAlgebra.mul!(y_test, B', x, true, false) 
                @test y_ref == y_test
                x[i] = false
            end
        end
    end
end
