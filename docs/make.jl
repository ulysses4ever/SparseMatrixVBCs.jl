using SparseMatrixVBCs
using Documenter

makedocs(;
    modules=[SparseMatrixVBCs],
    authors="Peter Ahrens",
    repo="https://github.com/peterahrens/SparseMatrixVBCs.jl/blob/{commit}{path}#L{line}",
    sitename="SparseMatrixVBCs.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://peterahrens.github.io/SparseMatrixVBCs.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/peterahrens/SparseMatrixVBCs.jl",
)
