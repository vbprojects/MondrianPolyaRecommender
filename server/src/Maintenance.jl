# One durable page per scheduler turn. The storage owner excludes all reads during
# the short publication/unlink step, so no external payload reader can race GC.
function reclamation_step!(ctx)
    s=ctx.store;day=string(floor(Int,s.clock/86400))
    DR.source_cursor(s,"__server_reclaimed_day")==day && return false
    raw=DR.source_cursor(s,"__server_reclaim_progress")
    state=isempty(raw) ? (phase="vectors",after=0,ceiling=s.next_block-1,day=day) : JSON3.read(raw)
    function progress(next)
        DR.set_cursor!(s,"__server_reclaim_progress",JSON3.write(next))
    end
    if state.phase=="vectors"
        blocks=DS.rows(s,"SELECT b.id FROM blocks b JOIN files f ON f.name=b.file WHERE b.live>0 AND b.id>? AND b.id<=? AND (b.live<b.n OR f.bytes>(SELECT COALESCE(SUM(other.bytes),0) FROM blocks other WHERE other.file=b.file AND other.live>0)) ORDER BY b.id LIMIT 1",(state.after,state.ceiling))
        DS.commit!(s) do
            if isempty(blocks)
                progress((phase="texts",after=0,ceiling=state.ceiling,day=state.day))
            else
                id=only(blocks).id
                refs=DS.rows(s,"SELECT id,row FROM posts WHERE status=1 AND block=? ORDER BY row",(id,))
                length(refs)<=256 || error("oversized frame")
                frame=DiskCodecs.subset(DS.frame_for(s,id),[p.row for p in refs])
                DS.replace_frame_refs!(s,frame,[p.id for p in refs])
                progress((phase="vectors",after=id,ceiling=state.ceiling,day=state.day))
            end
        end
    elseif state.phase=="texts"
        files=DS.rows(s,"SELECT f.name FROM files f WHERE f.kind='texts' AND EXISTS(SELECT 1 FROM posts p WHERE p.textfile=f.name AND p.status=1) AND f.bytes>(SELECT COALESCE(SUM(p.textlen),0) FROM posts p WHERE p.textfile=f.name AND p.status=1) ORDER BY f.name LIMIT 1")
        DS.commit!(s) do
            if isempty(files)
                progress((phase="gc",after=0,ceiling=state.ceiling,day=state.day))
            else
                name=String(only(files).name)
                ps=DS.rows(s,"SELECT id,textoff,textlen FROM posts WHERE status=1 AND textfile=? ORDER BY id LIMIT 128",(name,))
                open(joinpath(s.root,name)) do input
                    for p in ps
                        seek(input,p.textoff);data=read(input,p.textlen)
                        length(data)==p.textlen || error("short text read")
                        file,out=DS.packio(s,:texts);offset=position(out);write(out,data)
                        DS.exec(s,"UPDATE posts SET textfile=?,textoff=? WHERE id=?",(file,offset,p.id))
                    end
                end
            end
        end
    else
        files=DS.rows(s,"SELECT f.name FROM files f WHERE NOT EXISTS(SELECT 1 FROM blocks b WHERE b.file=f.name AND b.live>0) AND NOT EXISTS(SELECT 1 FROM posts p WHERE p.textfile=f.name AND p.status=1) LIMIT 32")
        # Unlink after commit. Crash orphan files are reclaimed on startup by the
        # existing store recovery; the next bounded cycle finishes referenced GC.
        DS.commit!(s) do
            for f in files
                DS.exec(s,"DELETE FROM blocks WHERE file=? AND live=0",(f.name,))
                DS.exec(s,"DELETE FROM files WHERE name=?",(f.name,))
            end
            if isempty(files)
                DR.set_cursor!(s,"__server_reclaimed_day",String(state.day))
                DR.set_cursor!(s,"__server_reclaim_progress","")
            end
        end
        for f in files;rm(joinpath(s.root,f.name);force=true);end
        !isempty(files) && DS.syncdir(s.root)
    end
    true
end
