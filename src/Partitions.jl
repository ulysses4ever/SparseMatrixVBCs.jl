struct Partition{Ti}
    spl::Vector{Ti}
    pos::Vector{Ti}
    ofs::Vector{Ti}
end

Base.length(b::Partition) = length(b.spl) - 1