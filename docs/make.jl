ENV["JULIA_PROGRESS"] = "false"

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
        size_threshold=nothing,
    ),
    pages=[
        "Home" => "index.md",
        "Theory" => "theory.md",
        "Tutorial" => "tutorial.md",
        "Citations" => "citations.md",
        "Reference" => "reference.md",
    ],
)

deploydocs(;
    repo="github.com/jfeldbrugge/Resurgence.jl",
    push_preview = true,
    devbranch="main",
)
