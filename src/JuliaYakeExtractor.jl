# Adapter only; the standalone implementation lives in julia-yake/.
include("../julia-yake/src/Yake.jl")
export JuliaYakeExtractor
struct JuliaYakeExtractor <: AbstractKeywordExtractor
    engine::Yake.Extractor
end
JuliaYakeExtractor()=JuliaYakeExtractor(Yake.Extractor(n=3,window_size=1,top=24,dedup_lim=0.9))
function extractor_id(e::JuliaYakeExtractor)
    # Reject mutated/custom engine settings: changing settings requires an explicit new ID.
    c=e.engine
    c.n==3 && c.window_size==1 && c.top==24 && c.dedup_lim==0.9 && c.stopwords==Yake.ENGLISH_STOPWORDS || error("unsupported Julia YAKE adapter configuration")
    "julia-yake@0.1.0;en;n=3;dedup=0.9;window=1;top=24;maxchars=8192;clean-v1;tokenizer-v1"
end
function extract_keywords(e::JuliaYakeExtractor,inputs)
    length(inputs)<=1024 || error("keyword batch exceeds 1024")
    id=extractor_id(e)
    map(inputs) do input
        truncated=length(input.text)>8192
        text=first(input.text,8192)
        text=replace(text,r"https?://\S+|www\.\S+"=>" ")
        text=replace(text,r"(?<!\w)@[\w.:-]+(?:@[\w.-]+)?"=>" ")
        text=replace(text,"#"=>"")
        lang=lowercase(input.language)
        eligible=isempty(lang) || first(split(replace(lang,'_'=>'-'),'-'))=="en"
        kws=eligible && !isempty(strip(text)) ? Yake.extract_keywords(e.engine,text) : Yake.Keyword[]
        KeywordResult([Candidate(k.phrase,i,k.score) for (i,k) in enumerate(kws)],id,truncated)
    end
end
