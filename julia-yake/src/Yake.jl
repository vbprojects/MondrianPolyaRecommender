# SPDX-License-Identifier: AGPL-3.0-or-later
# Julia adaptation of YAKE 0.6.0 (INESC TEC); see ../LICENSE and ../NOTICE.md.
module Yake
export Extractor, Keyword, extract_keywords, tokenize_sentences, score_tokens
const PUNCT = Set(Char.(vcat(collect(33:47),collect(58:64),collect(91:96),collect(123:126))))
Base.include_dependency(joinpath(@__DIR__,"../data/stopwords_en.txt"))
const ENGLISH_STOPWORDS = Set(lowercase.(split(read(joinpath(@__DIR__,"../data/stopwords_en.txt"),String),'\n')))
struct Extractor
    n::Int
    window_size::Int
    top::Int
    dedup_lim::Float64
    stopwords::Set{String}
    function Extractor(;n=3,window_size=1,top=20,dedup_lim=0.9,stopwords=ENGLISH_STOPWORDS)
        n>=1 && window_size>=1 && top>=0 && isfinite(dedup_lim) && 0<=dedup_lim<=1 || throw(ArgumentError("invalid YAKE configuration"))
        new(n,window_size,top,dedup_lim,Set(String.(stopwords)))
    end
end
struct Keyword
    phrase::String
    score::Float64
end
mutable struct Term
    stop::Bool
    tf::Int
    acronym::Int
    proper::Int
    sentences::Vector{Int}
    left::Dict{Int,Int}
    right::Dict{Int,Int}
    h::Float64
end
Term(stop)=Term(stop,0,0,0,Int[],Dict(),Dict(),1.)
mutable struct Phrase
    surface::String
    key::String
    ids::Vector{Int}
    tf::Int
    valid::Bool
    score::Float64
end
include("tokenizer.jl")
function tag(word,pos)
    stripped=replace(word,","=>"")
    if all(isdigit,stripped) || all(isdigit,replace(stripped,"."=>"";count=1))
        return 'd'
    end
    digits=count(isdigit,word);letters=count(isletter,word);punct=count(c->c in PUNCT,word)
    ((digits>0 && letters>0) || (digits==0 && letters==0) || punct>1) && return 'u'
    cased=filter(c->isuppercase(c)||islowercase(c),collect(word))
    !isempty(cased) && all(isuppercase,cased) && return 'a'
    length(word)>1 && isuppercase(first(word)) && pos>0 && count(isuppercase,word)==1 && return 'n'
    'p'
end
function similarity(a::Vector{Char},b::Vector{Char})
    isempty(a) && isempty(b) && return 1.
    prev=collect(0:length(b));row=similar(prev)
    for i in eachindex(a)
        row[1]=i
        for j in eachindex(b)
            row[j+1]=min(row[j]+1,prev[j+1]+1,prev[j]+(a[i]!=b[j]))
        end
        row,prev=prev,row
    end
    1-prev[end]/max(length(a),length(b))
end
"""YAKE scoring on supplied sentences/tokens; useful for tokenizer-independent parity tests."""
function score_tokens(e::Extractor,sentences)
    e.top==0 && return Keyword[]
    terms=Term[];ids=Dict{String,Int}();phrases=Phrase[];pids=Dict{String,Int}()
    for (sent,sentence) in enumerate(sentences)
        block=Tuple{Int,Char,String}[]
        for (pos,raw) in enumerate(sentence)
            word=String(raw)
            isempty(word) && continue
            if all(c->c in PUNCT,word);empty!(block);continue;end
            key=lowercase(word);wasstop=key in e.stopwords
            endswith(key,"s") && length(key)>3 && (key=chop(key))
            id=get(ids,key,0)
            if id==0
                stop=wasstop || key in e.stopwords || count(c->!(c in PUNCT),key)<3
                push!(terms,Term(stop));id=length(terms);ids[key]=id
            end
            t=terms[id];kind=tag(word,pos-1);t.tf+=1
            t.acronym+=(kind=='a');t.proper+=(kind=='n')
            (isempty(t.sentences) || last(t.sentences)!=sent-1) && push!(t.sentences,sent-1)
            if !(kind in ('u','d'))
                for j in max(1,length(block)-e.window_size+1):length(block)
                    left,ltag,_=block[j];ltag in ('u','d') && continue
                    terms[left].right[id]=get(terms[left].right,id,0)+1
                    t.left[left]=get(t.left,left,0)+1
                end
            end
            push!(block,(id,kind,word))
            for size in 1:min(e.n,length(block))
                group=@view block[end-size+1:end]
                surface=join((v[3] for v in group),' ');pkey=lowercase(surface)
                valid=all(v->!(v[2] in ('u','d')),group)
                pi=get(pids,pkey,0)
                if pi==0
                    push!(phrases,Phrase(surface,pkey,[v[1] for v in group],0,valid,0.))
                    pi=length(phrases);pids[pkey]=pi
                else
                    phrases[pi].valid|=valid
                end
                phrases[pi].tf+=1
            end
        end
    end
    valid_tf=[t.tf for t in terms if !t.stop]
    isempty(valid_tf) && return Keyword[]
    avg=sum(valid_tf)/length(valid_tf)
    deviation=sqrt(sum(x->(x-avg)^2,valid_tf)/length(valid_tf))
    maxtf=maximum(t.tf for t in terms)
    for t in terms
        lsum=sum(values(t.left);init=0);rsum=sum(values(t.right);init=0)
        rel=(0.5+(lsum==0 ? 0. : length(t.left)/lsum)*(t.tf/maxtf))+(0.5+(rsum==0 ? 0. : length(t.right)/rsum)*(t.tf/maxtf))
        freq=t.tf/(avg+deviation);spread=length(t.sentences)/length(sentences)
        case=max(t.acronym,t.proper)/(1+log(t.tf))
        ns=length(t.sentences);med=(t.sentences[(ns+1)÷2]+t.sentences[(ns+2)÷2])/2
        position=log(log(3+med))
        t.h=position*rel/(case+freq/rel+spread/rel)
    end
    valid=Int[]
    for (i,p) in enumerate(phrases)
        (!p.valid || terms[first(p.ids)].stop || terms[last(p.ids)].stop) && continue
        product=1.;total=0.
        for (j,id) in enumerate(p.ids)
            t=terms[id]
            if !t.stop
                product*=t.h;total+=t.h
            else
                left=j>1 ? get(terms[p.ids[j-1]].right,id,0)/terms[p.ids[j-1]].tf : 0.
                right=j<length(p.ids) ? get(t.right,p.ids[j+1],0)/terms[p.ids[j+1]].tf : 0.
                probability=left*right
                product*=2-probability;total-=1-probability
            end
        end
        p.score=product/((1+total)*p.tf);push!(valid,i)
    end
    # Insertion-order tie breaking matches Python's stable candidate ordering.
    sort!(valid;by=i->(phrases[i].score,i))
    result=Keyword[];accepted=Vector{Char}[]
    for i in valid
        p=phrases[i];chars=collect(p.key)
        if e.dedup_lim<1 && any(other->min(length(chars),length(other))/max(length(chars),length(other))>e.dedup_lim && similarity(chars,other)>e.dedup_lim,accepted)
            continue
        end
        push!(accepted,chars)
        push!(result,Keyword(e.dedup_lim>=1 ? p.key : p.surface,p.score))
        length(result)>=e.top && break
    end
    result
end
"""Extract ranked keywords. Lower scores indicate greater document-local importance."""
extract_keywords(e::Extractor,text::AbstractString)=score_tokens(e,tokenize_sentences(replace(String(text),'\n'=>' ')))
end
