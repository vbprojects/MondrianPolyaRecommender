using Yake
text=isempty(ARGS) ? "Julia makes scientific computing fast. Julia supports numerical computing and machine learning." : join(ARGS," ")
for keyword in extract_keywords(Extractor(n=3,top=8),text)
    println(keyword.phrase,"\t",keyword.score)
end
