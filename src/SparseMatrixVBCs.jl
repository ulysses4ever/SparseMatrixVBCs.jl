module SparseMatrixVBCs

using SparseArrays
using ChainPartitioners
using SIMD
using Hwloc
using BenchmarkTools
using Statistics
using LinearAlgebra
using MemoizedMethods
using DiskCaches
using Scratch
using SHA

using CpuId
const DEFAULT_SIMD_SIZE = CpuId.simdbytes()

export SparseMatrix1DVBC
export model_SparseMatrix1DVBC_blocks
export model_SparseMatrix1DVBC_memory
export model_SparseMatrix1DVBC_TrSpMV_time
export SparseMatrixVBC
export model_SparseMatrixVBC_blocks
export model_SparseMatrixVBC_memory
export model_SparseMatrixVBC_TrSpMV_time
export TrSpMV!

"""
    SparseMatrix1DVBC{W, Tv,Ti<:Integer} <: AbstractSparseMatrix{Tv,Ti}
Matrix type for storing sparse matrices in the
One-Dimensional Variable Block Column format.
"""
struct SparseMatrix1DVBC{W, Tv, Ti<:Integer} <: AbstractSparseMatrix{Tv,Ti}
    m::Int
    n::Int
    Φ::SplitPartition{Ti}
    pos::Vector{Ti}
    idx::Vector{Ti}
    ofs::Vector{Ti}
    val::Vector{Tv}
    function SparseMatrix1DVBC{W, Tv, Ti}(m::Integer, n::Integer, Φ::SplitPartition{Ti}, pos::Vector{Ti}, idx::Vector{Ti}, ofs::Vector{Ti}, val::Vector{Tv}) where {W, Tv, Ti<:Integer}
        @noinline throwsz(str, lbl, K) =
            throw(ArgumentError("number of $str ($lbl) must be ≥ 0, got $K"))
        m < 0 && throwsz("rows", 'm', m)
        n < 0 && throwsz("columns", 'n', n)
        W isa Int || throw(ArgumentError("W must be an Int"))
        W > 0 || throw(ArgumentError("W must be > 0"))
        new(m, n, Φ, pos, idx, ofs, val)
    end
end

Base.size(A::SparseMatrix1DVBC) = (A.m, A.n)

"""
    SparseMatrixVBC{W, Tv, Ti<:Integer} <: AbstractSparseMatrix{Tv,Ti}
Matrix type for storing sparse matrices in the
Variable Block Column format.
"""
struct SparseMatrixVBC{U, W, Tv, Ti<:Integer} <: AbstractSparseMatrix{Tv,Ti}
    m::Int
    n::Int
    Π::SplitPartition{Ti}
    Φ::SplitPartition{Ti}
    pos::Vector{Ti}
    idx::Vector{Ti}
    ofs::Vector{Ti}
    val::Vector{Tv}
    function SparseMatrixVBC{U, W, Tv, Ti}(m::Integer, n::Integer, Π::SplitPartition{Ti}, Φ::SplitPartition{Ti}, pos::Vector{Ti}, idx::Vector{Ti}, ofs::Vector{Ti}, val::Vector{Tv}) where {U, W, Tv, Ti<:Integer}
        @noinline throwsz(str, lbl, x) =
            throw(ArgumentError("number of $str ($lbl) must be ≥ 0, got $x"))
        m < 0 && throwsz("rows", 'm', m)
        n < 0 && throwsz("columns", 'n', n)
        U isa Int || throw(ArgumentError("U must be an Int"))
        W isa Int || throw(ArgumentError("W must be an Int"))
        U > 0 || throw(ArgumentError("U must be > 0"))
        W > 0 || throw(ArgumentError("W must be > 0"))
        new(m, n, Π, Φ, pos, idx, ofs, val)
    end
end

Base.size(A::SparseMatrixVBC) = (A.m, A.n)

include("util.jl")
include("TrSpMV.jl")
include("costs.jl")
include("constructors_1DVBC.jl")
include("constructors_VBC.jl")
include("hacks.jl")

end # module
