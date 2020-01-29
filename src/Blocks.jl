struct Blocks{Ti}
    spl::Vector{Ti}
    pos::Vector{Ti}
    ofs::Vector{Ti}
end

Base.length(b::Blocks) = length(b.spl) - 1