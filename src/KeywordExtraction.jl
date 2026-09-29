"""Extractor-independent keyword preparation. Storage consumes PreparedKeywords only."""
module KeywordExtraction
using JSON3, SHA, Unicode
export AbstractKeywordExtractor, KeywordInput, Candidate, KeywordResult, PreparedKeywords,
       PythonYakeExtractor, HashtagsOnlyExtractor, extract_keywords, extractor_id,
       prepare_keywords, finalize_keywords, fingerprint, input_digest, keyword_input, configured_extractor
abstract type AbstractKeywordExtractor end
struct KeywordInput
    identity::String
    text::String
    tags::Vector{String}
    language::String
end
struct Candidate
    phrase::String
    rank::Int
    score::Union{Nothing,Float64}
end
struct KeywordResult
    candidates::Vector{Candidate}
    extractor::String
    truncated::Bool
end
struct PreparedKeywords
    identity::String
    digest::String
    fingerprint::String
    terms::Vector{String}
    eligible::Bool
    assumed_english::Bool
    truncated::Bool
end
const POLICY="canonical-v1;nfkc-casefold;phrases=8;tags=4;phrase-words=3;maxchars=8192"
canonical(s)=strip(replace(strip(replace(Unicode.normalize(String(s);compat=true,casefold=true),r"\s+"=>" ")),r"^#+"=>""))
function keyword_input(p)
    m=JSON3.read(p.metadata)
    langs=get(m,:langs,String[])
    langs===nothing && (langs=String[])
    value=get(m,:language,isempty(langs) ? "" : first(langs))
    lang=value===nothing ? "" : String(value)
    KeywordInput("$(p.key.kind):$(p.key.native)",p.text,p.tags,lowercase(lang))
end
input_digest(x::KeywordInput)=bytes2hex(sha256(JSON3.write((x.identity,x.text,sort(unique(x.tags)),x.language))))
fingerprint(id::AbstractString)=bytes2hex(sha256(id*";"*POLICY))
extractor_id(e::AbstractKeywordExtractor)=error("implement extractor_id for $(typeof(e))")
fingerprint(e::AbstractKeywordExtractor)=fingerprint(extractor_id(e))
function finalize_keywords(input::KeywordInput,result::KeywordResult)
    eligible=isempty(input.language) || first(split(replace(lowercase(input.language),'_'=>'-'),'-'))=="en"
    terms=String[]
    if eligible
        for c in sort(result.candidates;by=c->(c.rank,c.phrase))
            t=canonical(c.phrase)
            isempty(t) && continue
            ncodeunits(t)<=256 && length(split(t))<=3 || continue
            t in terms || push!(terms,t)
            length(terms)>=8 && break
        end
    end
    tags=vcat(input.tags,[m.captures[1] for m in eachmatch(r"(?<![\p{L}\p{N}_])#([\p{L}\p{N}_]+)",input.text)])
    extra=0
    for tag in tags
        t=canonical(replace(tag,r"^#+"=>""))
        (isempty(t) || ncodeunits(t)>256 || t in terms) && continue
        push!(terms,t);extra+=1;extra>=4 && break
    end
    PreparedKeywords(input.identity,input_digest(input),fingerprint(result.extractor),terms,
        eligible,isempty(input.language),result.truncated)
end
function prepare_keywords(e::AbstractKeywordExtractor,inputs)
    length(inputs)<=1024 || error("keyword batch exceeds 1024")
    results=extract_keywords(e,inputs)
    length(results)==length(inputs) || error("extractor returned misaligned batch")
    all(r->r.extractor==extractor_id(e),results) || error("extractor version changed")
    [finalize_keywords(x,r) for (x,r) in zip(inputs,results)]
end
struct HashtagsOnlyExtractor <: AbstractKeywordExtractor end
extractor_id(::HashtagsOnlyExtractor)="hashtags-only-v1"
extract_keywords(e::HashtagsOnlyExtractor,inputs)=[KeywordResult(Candidate[],extractor_id(e),false) for _ in inputs]
mutable struct PythonYakeExtractor <: AbstractKeywordExtractor
    process::Base.Process
    id::String
    lock::ReentrantLock
end
function PythonYakeExtractor(;python=normpath(joinpath(@__DIR__,"../.venv-embeddings/bin/python")))
    p=open(Cmd([python,"-u",normpath(joinpath(@__DIR__,"../scripts/keyword_worker.py"))]),"r+")
    try
        ready=JSON3.read(readline(p));PythonYakeExtractor(p,String(ready.extractor),ReentrantLock())
    catch;close(p);rethrow();end
end
extractor_id(e::PythonYakeExtractor)=e.id
function extract_keywords(e::PythonYakeExtractor,inputs)
    isempty(inputs) && return KeywordResult[]
    length(inputs)<=1024 || error("keyword batch exceeds 1024")
    lock(e.lock) do
        write(e.process,JSON3.write((op="keywords",inputs=[(text=x.text,language=x.language) for x in inputs]))*"\n");flush(e.process)
        r=JSON3.read(readline(e.process));haskey(r,:error) && error("keyword worker: $(r.error)")
        String(r.extractor)==e.id || error("extractor version changed")
        [KeywordResult([Candidate(String(c.phrase),Int(c.rank),isnothing(c.score) ? nothing : Float64(c.score)) for c in q.candidates],e.id,Bool(q.truncated)) for q in r.results]
    end
end
Base.close(e::PythonYakeExtractor)=close(e.process)
include("JuliaYakeExtractor.jl")

"""Select preparation explicitly; omitted method preserves legacy store policy."""
function configured_extractor(cfg)
    method=get(cfg,"method","julia_yake")
    method=="hashtags" && return HashtagsOnlyExtractor()
    method=="julia_yake" && return JuliaYakeExtractor()
    throw(ArgumentError("keywords.method must be hashtags or julia_yake"))
end

end
