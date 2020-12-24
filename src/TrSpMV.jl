function TrSpMV!(y::Vector, A::SparseMatrixCSC, x::Vector)
    @fastmath @inbounds begin
        size(A, 2) == size(y, 1) || throw(DimensionMismatch())
        size(A, 1) == size(x, 1) || throw(DimensionMismatch())
        m = length(y)
        n = length(x)
        
        A_pos = A.colptr
        A_idx = A.rowval
        A_val = A.nzval
        for i = 1:length(y)
            tmp = zero(eltype(y))
            for j = A_pos[i]:(A_pos[i + 1] - 1)
                tmp += A_val[j] * x[A_idx[j]]
            end
            y[i] = tmp
        end
        y
    end
end

include("multiply_1DVBC.jl")
include("multiply_VBC.jl")