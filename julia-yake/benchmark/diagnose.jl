using JSON3
include("../src/Yake.jl")
using .Yake
cases=JSON3.read(read(joinpath(@__DIR__,"../test/python_reference.json"),String))
failures=Any[];token_exact=0;raw_exact=0
for (i,c) in enumerate(cases)
    e=Extractor(n=c.n,window_size=c.window,dedup_lim=c.dedup,top=24)
    a=score_tokens(e,[String.(s) for s in c.tokens]);b=[(String(k[1]),Float64(k[2])) for k in c.keywords]
    if !(length(a)==length(b) && all(x.phrase==y[1] && isapprox(x.score,y[2];rtol=1e-12,atol=1e-14) for (x,y) in zip(a,b)))
        push!(failures,(case=i,text=c.text,expected=b,actual=[(x.phrase,x.score) for x in a]))
    end
    global token_exact+=tokenize_sentences(replace(String(c.text),'\n'=>' '))==[String.(s) for s in c.tokens]
    raw=extract_keywords(e,String(c.text))
    global raw_exact+=length(raw)==length(b) && all(x.phrase==y[1] && isapprox(x.score,y[2];rtol=1e-12,atol=1e-14) for (x,y) in zip(raw,b))
end
open(io->JSON3.pretty(io,failures),joinpath(@__DIR__,"failures.json"),"w")
println("Scoring failures $(length(failures))/$(length(cases)); exact tokens $token_exact; exact end-to-end $raw_exact")
