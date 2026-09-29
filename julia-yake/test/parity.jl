using Test, JSON3
include("../src/Yake.jl")
using .Yake
cases=JSON3.read(read(joinpath(@__DIR__,"python_reference.json"),String))
score_failures=Any[];raw_exact=0;token_exact=0;max_error=0.
@testset "Python YAKE 0.6.0 scoring on identical tokens" begin
    for (i,c) in enumerate(cases)
        e=Extractor(n=c.n,window_size=c.window,dedup_lim=c.dedup,top=24)
        actual=score_tokens(e,[String.(s) for s in c.tokens])
        expected=[(String(k[1]),Float64(k[2])) for k in c.keywords]
        # Scores can round differently across NumPy/Julia: tolerate only tied ordering.
        byphrase=Dict(expected)
        matches=length(actual)==length(expected) && all(haskey(byphrase,a.phrase) && isapprox(a.score,byphrase[a.phrase];rtol=1e-12,atol=1e-14) for a in actual) &&
            all(isapprox(a.score,b[2];rtol=1e-12,atol=1e-14) for (a,b) in zip(actual,expected))
        if !matches
            push!(score_failures,(case=i,text=String(c.text),expected=expected,actual=[(x.phrase,x.score) for x in actual]))
        end
        @test matches
        if length(actual)==length(expected)
            for (a,b) in zip(actual,expected);global max_error=max(max_error,abs(a.score-b[2]));end
        end
        global token_exact+=tokenize_sentences(replace(String(c.text),'\n'=>' '))==[String.(s) for s in c.tokens]
        raw=extract_keywords(e,String(c.text))
        global raw_exact+=(length(raw)==length(expected) && all(a.phrase==b[1] && isapprox(a.score,b[2];rtol=1e-12,atol=1e-14) for (a,b) in zip(raw,expected)))
    end
end
println("token_exact=$token_exact raw_exact=$raw_exact cases=$(length(cases)) max_error=$max_error")
