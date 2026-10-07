using Resurgence
using Documenter

DocMeta.setdocmeta!(Resurgence, :DocTestSetup, :(using Resurgence); recursive=true)

makedocs(;
    modules=[Resurgence],
    authors="Job Feldbrugge <14946916+jfeldbrugge@users.noreply.github.com> and contributors",
    sitename="Resurgence.jl",
    format=Documenter.HTML(;
        canonical="https://jfeldbrugge.github.io/Resurgence.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/jfeldbrugge/Resurgence.jl",
    devbranch="main",
)
