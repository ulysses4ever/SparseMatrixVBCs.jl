module SparseMatrix1DVBCs

using SparseArrays
using ChainPartitioners
using SIMD
using Hwloc
using BenchmarkTools
using BSON
using Statistics

export model_SparseMatrix1DVBC_blocks
export model_SparseMatrix1DVBC_memory
export model_SparseMatrix1DVBC_time
export SparseMatrix1DVBC
export TrSpMV!

"""
    SparseMatrix1DVBC{Ws, Tv,Ti<:Integer} <: AbstractSparseMatrix{Tv,Ti}
Matrix type for storing sparse matrices in the
Variably Compressed Column Compressed Sparse Column format.
"""
struct SparseMatrix1DVBC{Ws, Tv, Ti<:Integer} <: AbstractSparseMatrix{Tv,Ti}
    m::Int
    n::Int
    spl::Vector{Ti}
    pos::Vector{Ti}
    idx::Vector{Ti}
    ofs::Vector{Ti}
    val::Vector{Tv}
    function SparseMatrix1DVBC{Ws, Tv, Ti}(m::Integer, n::Integer, spl::Vector{Ti}, pos::Vector{Ti}, idx::Vector{Ti}, ofs::Vector{Ti}, val::Vector{Tv}) where {Ws, Tv, Ti<:Integer}
        @noinline throwsz(str, lbl, K) =
            throw(ArgumentError("number of $str ($lbl) must be ≥ 0, got $K"))
        m < 0 && throwsz("rows", 'm', m)
        n < 0 && throwsz("columns", 'n', n)
        Ws isa Tuple{Vararg{Int}} || throw(ArgumentError("Ws must be a tuple of integers"))
        minimum(Ws) > 0 || throw(ArgumentError("Ws must be > 0"))
        new(m, n, spl, pos, idx, ofs, val)
    end
end

Base.size(A::SparseMatrix1DVBC) = (A.m, A.n)

const cachefile = joinpath(@__DIR__(), "cache.bson")

include("util.jl")
include("TrSpMV.jl")
include("costs.jl")
include("constructors.jl")

end # module