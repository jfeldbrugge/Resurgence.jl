using GaussManinResurgence
using Documenter

DocMeta.setdocmeta!(GaussManinResurgence, :DocTestSetup, :(using GaussManinResurgence); recursive=true)

makedocs(;
    modules=[GaussManinResurgence],
    authors="Job Feldbrugge <14946916+jfeldbrugge@users.noreply.github.com> and contributors",
    sitename="GaussManinResurgence.jl",
    format=Documenter.HTML(;
        canonical="https://jfeldbrugge.github.io/GaussManinResurgence.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/jfeldbrugge/GaussManinResurgence.jl",
    devbranch="main",
)
