function json(io,x::Real)
    isfinite(x) ? print(io,x) : print(io,"null")
end
json(io,x::AbstractString) = print(io,'"',replace(x,"\\"=>"\\\\","\""=>"\\\"","\n"=>"\\n"),'"')
function json(io,x::Union{AbstractVector,Tuple})
    print(io,'[')
    for (i,v) in enumerate(x)
        i>1 && print(io,','); json(io,v)
    end
    print(io,']')
end
function json(io,x::NamedTuple)
    print(io,'{')
    for (i,k) in enumerate(keys(x))
        i>1 && print(io,','); json(io,string(k)); print(io,':'); json(io,x[k])
    end
    print(io,'}')
end

