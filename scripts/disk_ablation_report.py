#!/usr/bin/env python3
"""Validate and summarize the disk ablation; HTML needs no server or network."""
import csv, hashlib, json, math, pathlib, sqlite3, statistics as st
ROOT=pathlib.Path(__file__).resolve().parents[1]
OUT=ROOT/"output/disk_ablation"
MODES=["f32","f16","q8","centroid_shared","centroid_leaf","pca16"]
LABELS=dict(zip(MODES,["Float32","Float16","Per-vector Int8","Current centroid Int8","Leaf-local centroid Int8","Frozen leaf PCA r16"]))
mean=lambda xs: st.mean(list(xs))
def quantile(xs,p):
    xs=sorted(xs)
    if not xs:return 0.
    at=(len(xs)-1)*p;a=int(at);b=min(a+1,len(xs)-1)
    return xs[a]+(at-a)*(xs[b]-xs[a])
def read(name):return json.loads((OUT/name).read_text())
def summarize_queries(qs):
    return dict(n=len(qs),p50=quantile([q["ms"] for q in qs],.5),p95=quantile([q["ms"] for q in qs],.95),
        p99=quantile([q["ms"] for q in qs],.99),blocks=mean(q["blocks"] for q in qs),decoders=mean(q["decoders"] for q in qs),
        app_bytes_per_post=sum(q["application_read_bytes"] for q in qs)/max(1,sum(q["returned"] for q in qs)),
        disk_bytes_per_post=sum(q["os_read_bytes"] for q in qs)/max(1,sum(q["returned"] for q in qs)),
        underfilled=sum(q["underfilled"] for q in qs),returned=mean(q["returned"] for q in qs),rss_mib=mean(q["rss"] for q in qs)/2**20,
        hit_rate=sum(q["cache_hits"] for q in qs)/max(1,sum(q["cache_hits"]+q["cache_misses"] for q in qs)),
        sqlite_hits=sum(q.get("sqlite_hits",0) for q in qs),sqlite_misses=sum(q.get("sqlite_misses",0) for q in qs))
def main():
    raw=[read(pathlib.Path("raw")/f.name) for f in sorted((OUT/"raw").glob("*.json"))]
    quality=[read(pathlib.Path("quality")/f.name) for f in sorted((OUT/"quality").glob("*.json"))]
    stress=read("stress.json");updates=read("budget_updates.json");cold=read("cold_io.json");traffic=read("traffic.json")
    assert len(raw)==36 and len(quality)==18 and len(stress)==72 and len(updates)==144 and len(cold)==144
    assert len(traffic)==144 and all(q.get("protocol_version")==3 for q in quality)
    assert all(len(r["queries"])==4 and r["count"]==100000 for r in raw)
    keys={(r["seed"],r["mode"],r["layout"]) for r in raw};assert len(keys)==36
    for seed in [17,29,43]:
        cases=[r for r in raw if r["seed"]==seed]
        assert all(r["tree"]==cases[0]["tree"] for r in cases), "formats changed the common tree"
        for mode in MODES:
            pair=[r for r in cases if r["mode"]==mode]
            assert pair[0]["fingerprint"]==pair[1]["fingerprint"], "layout changed encoded values"
    assert all(r["correctness"]["relocation_exact"] and r["correctness"]["audited_posts"]==100000 for r in raw)
    for r in raw:
        if r["layout"]!="leaf":continue
        path=ROOT/"data/disk_ablation"/r["name"]/"index.sqlite"
        with sqlite3.connect(f"file:{path}?mode=ro",uri=True) as db:
            assert db.execute("SELECT COUNT(*) FROM (SELECT block FROM posts WHERE status=1 GROUP BY block HAVING COUNT(DISTINCT leaf)>1)").fetchone()[0]==0,"physical leaf block spans leaves"
    assert all(s["exact_relocation"] and s["replay_ignored"]==1 for s in stress)
    for seed in [17,29,43]:
        for dimension in [2,32]:
            cases=[s for s in stress if s["seed"]==seed and s["dimension"]==dimension]
            for phase in ["timeline","daily"]:
                shapes=[[t["tree"] for t in s[phase]] for s in cases]
                assert all(x==shapes[0] for x in shapes),"synthetic formats did not share structural fixtures"
    assert all(q["restart_empty"] for r in raw for q in r["queries"])
    assert all(q["stats"]["block_evictions"]>0 for r in raw for q in r["queries"] if q["budget"]==32*2**20)
    assert all(q["stats"]["state_cache_peak"]+q["stats"]["block_cache_peak"]+q["stats"]["sqlite_cache_budget"]+q["stats"]["workspace_reservation"]<=q["budget"] for r in raw for q in r["queries"])
    assert all(q["sqlite_bytes"]<=b["stats"]["sqlite_cache_budget"] for r in raw for b in r["queries"] for q in b["queries"])
    assert all(q["recall10"]==1 and q["cut_disagreement"]==0 for q in quality if q["mode"]=="f32")
    summary=[]
    for mode in MODES:
        qq=[q for q in quality if q["mode"]==mode]
        for layout in ["append","leaf"]:
            rr=[r for r in raw if r["mode"]==mode and r["layout"]==layout]
            item=dict(mode=mode,label=LABELS[mode],layout=layout,seeds=3,
                total_mib=mean(r["storage"]["total_bytes"] for r in rr)/2**20,
                vector_bpp=mean(r["storage"]["vector_bytes"]/r["count"] for r in rr),
                decoder_bpp=mean(r["storage"]["decoder_bytes"]/r["count"] for r in rr),
                text_mib=mean(r["storage"]["text_files"] for r in rr)/2**20,
                sqlite_mib=mean(r["storage"]["sqlite_bytes"] for r in rr)/2**20,
                identity_mib=mean(r["db_breakdown"].get("identities",0) for r in rr)/2**20,
                post_catalog_mib=mean(r["db_breakdown"].get("posts",0) for r in rr)/2**20,
                density_state_mib=mean(r["db_breakdown"].get("states",0) for r in rr)/2**20,
                frame_mib=mean(r["storage"]["vector_files"] for r in rr)/2**20,
                fit_occupancy=mean(r["metrics"]["encoded_posts"]/r["metrics"]["encoded_groups"] for r in rr),
                stored_frame_occupancy=mean(r["count"]/r["storage"]["blocks"] for r in rr),
                fit_rank=mean(r["metrics"]["encoded_rank_sum"]/r["metrics"]["encoded_posts"] for r in rr),
                ingest_pps=mean(r["count"]/r["ingestion"]["seconds"] for r in rr),
                ingest_seed_min=min(r["count"]/r["ingestion"]["seconds"] for r in rr),
                commit_api_pps=mean(r["count"]/r["ingestion"]["durable_seconds"] for r in rr),
                split_ms=mean(1000*r["metrics"]["split_seconds"]/r["metrics"]["splits"] for r in rr),
                compaction_seconds=mean(r["compaction"]["seconds"] for r in rr),
                compaction_mib=mean(r["compaction"]["bytes"] for r in rr)/2**20,
                peak_compaction_vector_files_mib=mean(r["before_compaction"]["vector_files"]+r["compaction"]["bytes"] for r in rr)/2**20,
                peak_rss_mib=mean(r["ingestion"]["memory"]["peak_rss"] for r in rr)/2**20,
                recall=mean(q["recall10"] for q in qq),recall_min=min(q["recall10"] for q in qq),recall_max=max(q["recall10"] for q in qq),
                tie_recall=mean(q["tie_aware_recall10"] for q in qq),cosine_error=mean(q["cosine_error_mean"] for q in qq),
                cut_error=mean(q["cut_disagreement"] for q in qq),ig_agreement=mean(q["ig_agreement"] for q in qq),ig_regret=mean(q["ig_regret_bits"] for q in qq),
                regional={str(b):mean(t["preservation"] for q in qq for t in q["regional"] if t["leaf_budget"]==b) for b in [1,4,16]},
                global_regional={str(b):mean(t["global_recall"] for q in qq for t in q["regional"] if t["leaf_budget"]==b) for b in [1,4,16]},
                reference_regional={str(b):mean(t["float32_global_recall"] for q in qq for t in q["regional"] if t["leaf_budget"]==b) for b in [1,4,16]},
                budget={})
            item["weekly_gib_estimate"]=item["total_mib"]*604.8/1024
            synthetic=[s for s in stress if s["mode"]==mode and s["layout"]==layout and s["dimension"]==32]
            item["synthetic_merge_seconds"]=mean(t["merge_seconds"] for s in synthetic for t in s["timeline"])
            item["synthetic_expired_posts"]=sum(t["expired"] for s in synthetic for t in s["daily"])
            for budget in [32,256,1024,4096]:
                qr=[z for r in rr for z in r["queries"] if z["budget"]==budget*2**20]
                qs=[z for r in qr for z in r["queries"]]
                cc=[z for r in cold if r["mode"]==mode and r["layout"]==layout and r["budget_mib"]==budget for z in r["queries"]]
                cc += [z for r in traffic if r["mode"]==mode and r["layout"]==layout and r["budget_mib"]==budget for z in r["queries"]]
                bb=[u for u in updates if u["mode"]==mode and u["layout"]==layout and u["budget_mib"]==budget]
                cells={}
                for condition in ["application_cold","immediate_repeat","mixed","file_cache_eviction_hint","continuous_uniform","continuous_topic_burst","continuous_cold_revisit"]:
                    for b in [1,4,16]:
                        selected=[q for q in qs+cc if q["condition"]==condition and q["leaf_budget"]==b]
                        if selected:cells[f"{condition}:{b}"]=summarize_queries(selected)
                item["budget"][str(budget)]=dict(cells=cells,
                    block_evictions=mean(q["stats"]["block_evictions"] for q in qr),state_evictions=mean(q["stats"]["state_evictions"] for q in qr),
                    observed_sqlite_mib=mean(q["stats"]["sqlite_actual"]["cache_bytes"] for q in qr)/2**20,
                    update_pps=mean(u["posts_per_second"] for u in bb),
                    process_peak_mib=max(q["memory"]["peak_rss"] for q in qr)/2**20)
            summary.append(item)
    checks=dict(real_stores=36,quality_audits=18,synthetic_cases=72,cold_query_runs=144,budget_update_probes=144,
        exact_layout_pairs=18,real_posts_audited=3600000,common_tree=True,relocation_exact=True,
        continuous_traffic_runs=len(traffic),
        measured_queries=sum(len(b["queries"]) for r in raw for b in r["queries"])+sum(len(r["queries"]) for r in cold+traffic),
        identity_lookup_checks=sum(len(r["identity_lookups"]) for r in traffic),
        synthetic_splits=sum(s["metrics"].get("splits",0) for s in stress),synthetic_merges=sum(s["metrics"].get("merges",0) for s in stress),
        actual_cold_reads=any(q["os_read_bytes"]>0 for r in cold for q in r["queries"]))
    result=dict(config=read("config.json"),summary=summary,checks=checks,raw=raw,quality=quality,stress=stress,budget_updates=updates,cold_io=cold,traffic=traffic)
    (OUT/"results.json").write_text(json.dumps(result,separators=(",",":"))+"\n")
    (OUT/"summary.json").write_text(json.dumps(dict(summary=summary,checks=checks),indent=2)+"\n")
    with (OUT/"summary.csv").open("w") as f:
        fields=[k for k in summary[0] if k not in ["budget","regional","global_regional","reference_regional"]]
        w=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore");w.writeheader();w.writerows(summary)
    with (OUT/"query_measurements.csv").open("w") as f:
        records=[]
        for r in raw:
            for b in r["queries"]:
                for q in b["queries"]:records.append(dict(seed=r["seed"],mode=r["mode"],layout=r["layout"],budget_mib=b["budget"]//2**20,**q))
        for b in cold+traffic:
            for q in b["queries"]:records.append(dict(seed=b["seed"],mode=b["mode"],layout=b["layout"],budget_mib=b["budget_mib"],**q))
        fields=list(dict.fromkeys(k for r in records for k in r));w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(records)
    recommendation=("Recommend Float16 for the first leaf-local version: its cosine fidelity is nearly exact and it needs no shared decoder. "
        "Per-vector Int8 is the smaller independently decodable alternative, with more routing disagreement. "
        "The existing centroid/append control is also competitive on space and quality if cross-leaf packing remains acceptable. "
        "Frozen local PCA does not justify its decoder cost under these small-group rules.")
    fa=next(r for r in summary if r["mode"]=="f16" and r["layout"]=="append")
    fl=next(r for r in summary if r["mode"]=="f16" and r["layout"]=="leaf")
    ca=fa["budget"]["256"]["cells"]["file_cache_eviction_hint:4"]
    cl=fl["budget"]["256"]["cells"]["file_cache_eviction_hint:4"]
    if cl["p95"]<ca["p95"] and cl["disk_bytes_per_post"]<ca["disk_bytes_per_post"]:
        layout_note=(f'For Float16, leaf packing is worthwhile for regional reads in this test: '
            f'cold p95 is {cl["p95"]:.2f} versus {ca["p95"]:.2f} ms at four leaves/256 MiB. '
            f'Linux reads fall from {ca["disk_bytes_per_post"]/1024:.0f} to {cl["disk_bytes_per_post"]/1024:.0f} KiB/post. '
            'This supports leaf packing for the intended cold, disk-resident workload, not a universal layout win.')
    else:
        layout_note=('The measured Float16 cold-read latency and I/O do not jointly establish a leaf-packing win. '
            'Keep append layout as the simpler default and use the table to choose for the intended workload.')
    wa=fa["budget"]["256"]["cells"]["continuous_uniform:4"]
    wl=fl["budget"]["256"]["cells"]["continuous_uniform:4"]
    layout_note+=(f' Continuous uniform-traffic p95 is {wa["p95"]:.2f} ms for append and {wl["p95"]:.2f} ms for leaf; '
        f'compaction takes {fa["compaction_seconds"]:.2f} versus {fl["compaction_seconds"]:.2f} seconds. '
        'Retain append as the hot-cache/write-heavy option, and optimize the leaf compactor before larger tests. '
        'Weekly-scale behavior is unvalidated.')
    recommendation+=' '+layout_note
    lines=["Leaf-local disk ablation: 100,000 GTE-tiny posts × 3 seeds",recommendation,"",
        "Format / layout | total MiB | vector+decoder B/post | recall@10 | leaf assignment disagreement | durable posts/s | cold p95 ms (4 leaves,256 MiB)"]
    for r in summary:
        coldcell=r["budget"]["256"]["cells"]["file_cache_eviction_hint:4"]
        lines.append(f'{r["label"]} / {r["layout"]} | {r["total_mib"]:.2f} | {r["vector_bpp"]+r["decoder_bpp"]:.2f} | {r["recall"]:.4f} | {r["cut_error"]:.4%} | {r["ingest_pps"]:.0f} | {coldcell["p95"]:.3f}')
    lines += ["", "Budgets bound accounted caches plus a 16 MiB workspace reservation; process RSS is reported separately.",
        "Ingestion above uses 256 MiB. Every other budget has a durable 1,024-post update probe on a disposable snapshot.",
        "Cold-file eviction is advisory; /proc/self/io read_bytes records Linux block reads including readahead. WSL2 host/device caches remain uncontrolled.",
        "P99 is descriptive for these small query samples; it is not a production tail-latency guarantee.",
        "Weekly sizes scale measured bytes/post to 60,480,000 active posts; they are estimates, not weekly-scale validation.",
        "Identity tombstones and SQLite free pages persist; metadata retention and vacuum policy remain production work.",
        "Information gain uses the existing entropy/response functions with synthetic cosine-distance feedback, not real user preferences.",
        json.dumps(checks,sort_keys=True)]
    (OUT/"summary.txt").write_text("\n".join(lines)+"\n")
    data=json.dumps(dict(summary=summary,checks=checks,recommendation=recommendation),separators=(",",":")).replace("<","\\u003c")
    html=(ROOT/"examples/disk_ablation_viewer.html").read_text().replace("__RESULTS__",data)
    (OUT/"demo.html").write_text(html)
    print("\n".join(lines))
if __name__=="__main__":main()
