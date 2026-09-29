# SPDX-License-Identifier: AGPL-3.0-or-later
# Word-boundary rules adapted from segtok 1.5.11 (MIT; Florian Leitner).
# Sentence segmentation is a native conservative approximation, not a segtok port.
const ALNUM = raw"[\p{Ll}\p{Lm}\p{Lt}\p{Lu}\p{Nd}\p{Nl}]"
const APO = "[´ʹʼ’′]"
const HYPHEN = "[\u00ad\u058a\u05be\u0f0c\u1400\u1806\u2010-\u2012\u2e17\u30a0-]"
const WORD = Regex("(?:" * ALNUM * raw"\.(?!\.\.)|" * ALNUM * ",(?=" * ALNUM * ")|[\\p{Nd}\\p{Nl}]:(?=[\\p{Nd}\\p{Nl}])|" * ALNUM * APO * "?" * HYPHEN * "(?=" * ALNUM * ")|" * APO * "(?!" * APO * ")|" * ALNUM * "'(?=" * ALNUM * ")|s'\$|" * ALNUM * ")+")
const WORD_START=Regex("^"*WORD.pattern)
const APO_START=Regex("^"*APO)
const TERMINALS = Set(collect(".!?‼‽⁇⁈⁉。﹒﹖﹗！．？"))
const ABBREVIATIONS = Set(split("dr mr mrs ms prof sr jr st vs etc e.g i.e fig no dept inc ltd approx vol pp al mt rev gen sen rep"))
function sentence_segments(text)
    out=String[];start=firstindex(text)
    for m in eachmatch(r"[.!?‼‽⁇⁈⁉。﹒﹖﹗！．？]+[\"'’”\)\]]*(?:\s+|$)",text)
        stop=prevind(text,m.offset+ncodeunits(m.match))
        fragment=String(SubString(text,start,stop));stripped=strip(fragment)
        lastword=isempty(stripped) ? "" : last(split(stripped))
        abbreviation=endswith(stripped,".") && (lowercase(chop(lastword)) in ABBREVIATIONS || occursin(r"^(?:[A-Za-z]\.)+$",lastword))
        abbreviation && stop<lastindex(text) && continue
        !isempty(stripped) && push!(out,String(stripped))
        start=nextind(text,stop)
    end
    start<=lastindex(text) && !isempty(strip(SubString(text,start))) && push!(out,String(strip(SubString(text,start))))
    out
end
const ENTITIES=Dict("amp"=>"&","lt"=>"<","gt"=>">","quot"=>"\"","apos"=>"'","nbsp"=>" ")
function unescape_html(text)
    replace(text,r"&(?:#x[0-9a-fA-F]+|#[0-9]+|[A-Za-z]+);"=>x->begin
        entity=String(x[2:end-1])
        if startswith(entity,"#")
            n=tryparse(UInt32,startswith(entity,"#x") ? entity[3:end] : entity[2:end];base=startswith(entity,"#x") ? 16 : 10)
            n!==nothing && n<=0x10ffff && !(0xd800<=n<=0xdfff) ? string(Char(n)) : "�"
        else
            get(ENTITIES,entity,String(x))
        end
    end)
end
function word_tokens(sentence)
    tokens=String[]
    for span in split(unescape_html(sentence))
        start=firstindex(span)
        for m in eachmatch(WORD,span)
            m.offset>start && push!(tokens,String(SubString(span,start,prevind(span,m.offset))))
            push!(tokens,String(m.match));start=m.offset+ncodeunits(m.match)
        end
        start<=lastindex(span) && push!(tokens,String(SubString(span,start)))
    end
    for i in length(tokens):-1:max(1,length(tokens)-2)
        word=tokens[i]
        if (occursin(WORD_START,word) && !occursin(APO_START,word)) || any(c->c in TERMINALS,word)
            if length(word)>1 && word!="..."
                if last(word) in TERMINALS
                    tokens[i]=chop(word);insert!(tokens,i+1,string(last(word)))
                elseif first(word) in TERMINALS
                    tokens[i]=string(first(word));insert!(tokens,i+1,String(SubString(word,nextind(word,1))))
                end
            end
            break
        end
    end
    # Dangling commas/colons and English contractions; retain non-ASCII suffixes
    # just as YAKE's Python token filter does (it filters only ASCII apostrophes).
    out=String[]
    for word in tokens
        tail=String[]
        while length(word)>1 && last(word) in (',',';',':')
            pushfirst!(tail,string(last(word)));word=chop(word)
        end
        match_=match(r"^(.+)(['´ʹʼ’′](?:d|ll|m|re|s|t|ve))$",word)
        if match_!==nothing
            prefix=String(match_[1]);suffix=String(match_[2])
            if last(suffix)=='t' && endswith(prefix,"n")
                prefix=chop(prefix);suffix="n"*suffix
            end
            !isempty(prefix) && push!(out,prefix)
            !startswith(suffix,"'") && push!(out,suffix)
        elseif !(startswith(word,"'") && length(word)>1) && !isempty(word)
            push!(out,word)
        end
        append!(out,tail)
    end
    out
end
"""Native Unicode tokenization. Sentence-boundary differences from segtok are measured separately."""
tokenize_sentences(text::AbstractString)=[word_tokens(s) for s in sentence_segments(replace(String(text),'\t'=>' '))]
