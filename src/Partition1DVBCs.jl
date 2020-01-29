struct Partition1DVBC{Ti}
    spl::Vector{Ti}
    pos::Vector{Ti}
    ofs::Vector{Ti}
end

Base.length(b::Partition1DVBC) = length(b.spl) - 1