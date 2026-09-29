using Test, Yake
@testset "Standalone Julia YAKE" begin
    e=Extractor()
    @test isempty(extract_keywords(e,""))
    @test isempty(extract_keywords(e,"the and or !!!"))
    @test isempty(extract_keywords(Extractor(top=0),"Julia language"))
    @test_throws ArgumentError Extractor(n=0)
    @test_throws ArgumentError Extractor(window_size=0)
    @test_throws ArgumentError Extractor(dedup_lim=NaN)
    @test_throws ArgumentError Extractor(top=-1)
    # One non-stopword: casing=0, frequency=1, spread=1, relatedness=1.
    h=log(log(3))/2
    k=only(score_tokens(e,[["galaxy"]]))
    @test k.phrase=="galaxy"
    @test k.score≈h/(1+h)
    @test Yake.similarity(collect("abc"),collect("abc"))==1
    @test Yake.similarity(collect("cat"),collect("cut"))≈2/3
    @test Yake.similarity(Char[],Char[])==1
    @test tokenize_sentences("Hello world. More stars!")==[["Hello","world","."],["More","stars","!"]]
    @test tokenize_sentences("I can't say it's bad.")==[["I","ca","n't","say","it","bad","."]]
    for text in ["NASA builds rockets. NASA rockets reach space.","café naïve résumé","你好世界。こんにちは！","🦋","Julia &amp; Python","one\ntwo","👨‍👩‍👧‍👦 space"]
        a=extract_keywords(e,text);b=extract_keywords(e,text)
        @test [(x.phrase,x.score) for x in a]==[(x.phrase,x.score) for x in b]
        @test issorted(getfield.(a,:score))
        @test all(x->isfinite(x.score),a)
        @test length(a)<=e.top
    end
    @test all(x->length(split(x.phrase))==1,extract_keywords(Extractor(n=1),"Space rockets explore distant galaxies"))
    @test isempty(extract_keywords(Extractor(stopwords=["galaxy"]),"galaxy"))
end
