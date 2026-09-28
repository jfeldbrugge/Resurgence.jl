using GaussManin
using Documenter

DocMeta.setdocmeta!(GaussManin, :DocTestSetup, :(using GaussManin); recursive=true)

makedocs(;
    modules=[GaussManin],
    authors="Job Feldbrugge <14946916+jfeldbrugge@users.noreply.github.com> and contributors",
    sitename="GaussManin.jl",
    format=Documenter.HTML(;
        canonical="https://jfeldbrugge.github.io/GaussManin.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/jfeldbrugge/GaussManin.jl",
    devbranch="main",
)
