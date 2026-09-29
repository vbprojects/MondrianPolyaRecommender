struct Job
    work::Function
    reply::Channel{Any}
    deadline::Float64
end
mutable struct Owner
    queue::Channel{Job}
    task::Task
    gate::ReentrantLock
    pending::Int
    capacity::Int
    accepting::Bool
    store::DS.Store
end
struct JobResult
    value::Any
    failed::Bool
end
function Owner(ctx)
    queue=Channel{Job}(ctx.settings.queue_capacity);gate=ReentrantLock()
    owner=Owner(queue,Task(()->nothing),gate,0,ctx.settings.queue_capacity,true,ctx.store)
    owner.task=Threads.@spawn begin
        for job in queue
            result=try
                time()>job.deadline && fail("request_timeout",503)
                JobResult(Base.invokelatest(job.work,ctx),false)
            catch e
                JobResult(e,true)
            end
            put!(job.reply,result)
            lock(gate) do;owner.pending-=1;end
        end
    end
    owner
end
function submit(f::Function,o::Owner;timeout=5.)
    reply=Channel{Any}(1);deadline=isfinite(timeout) ? time()+timeout : Inf
    lock(o.gate) do
        o.accepting || fail("server_stopping",503)
        o.pending<o.capacity || fail("queue_full",429)
        o.pending+=1
        put!(o.queue,Job(f,reply,deadline))
    end
    if isfinite(timeout)
        timedwait(()->isready(reply),timeout;pollint=.005)==:timed_out && fail("request_timeout",503)
    end
    r=take!(reply)
    r.failed && throw(r.value)
    r.value
end
function stop_owner!(o)
    lock(o.gate) do;o.accepting=false;close(o.queue);end
    wait(o.task)
end
