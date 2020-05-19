struct Partition{Ti}
    Π::Vector{Ti}
    pos::Vector{Ti}
    ofs::Vector{Ti}
end

Base.length(b::Partition) = length(b.Π) - 1