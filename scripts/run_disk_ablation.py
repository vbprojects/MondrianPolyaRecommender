#!/usr/bin/env python3
"""Resume the fixed three-seed experiment. Each case is a fresh Julia process.

Immutable experiment stores are in data/disk_ablation, never the live index.
Build jobs are serial: overlapping disk benchmarks would bias the results.
"""
import argparse, hashlib, json, os, pathlib, subprocess, sys, time
ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "output/disk_ablation"
FORMATS = ["f32", "f16", "q8", "centroid_shared", "centroid_leaf", "pca16"]
SEEDS = [17, 29, 43]

def main():
    p=argparse.ArgumentParser();p.add_argument("--quality-only",action="store_true");p.add_argument("--skip-followups",action="store_true");args=p.parse_args()
    OUT.mkdir(parents=True,exist_ok=True);(OUT/"logs").mkdir(exist_ok=True)
    config = dict(seeds=SEEDS, formats=FORMATS, layouts=["append","leaf"], posts=100000,
        microbatch=1024, codec_group_limit=256, dimension=384, ppca_rank=4,
        max_leaves=128, split_after=256, reservoir=64, half_life_seconds=21600,
        retention_seconds=604800, simulated_arrivals_per_second=100,
        budgets_mib=[32,256,1024,4096], ingestion_budget_mib=256,
        cold_condition="application caches cleared; OS page cache uncontrolled and read_bytes measured",
        reference_evaluation="separate process", sqlite="WAL synchronous FULL; one writer; immutable payload fsync",
        model_manifest=json.loads((ROOT/"data/gte_disk_ablation/manifest.json").read_text()))
    config["sources"]={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in
        [ROOT/"src/DiskTreeStore.jl",ROOT/"src/DiskCodecs.jl",ROOT/"src/LeafLatentTree.jl",
         ROOT/"examples/disk_ablation_worker.jl",ROOT/"examples/disk_ablation_common.jl",ROOT/"examples/disk_ablation_quality.jl",
         ROOT/"environments/storage/Manifest.toml"]}
    (OUT/"config.json").write_text(json.dumps(config,indent=2)+"\n")
    def run(script, argv, name):
        log=OUT/"logs"/(name+".log");print(time.strftime("%H:%M:%S"),name,flush=True)
        with log.open("w") as out:
            subprocess.run(["julia","--project=environments/storage",script,*argv],cwd=ROOT,
                stdout=out,stderr=subprocess.STDOUT,check=True,env={**os.environ,"OPENBLAS_NUM_THREADS":"1"})
    if not args.quality_only:
        for seed in SEEDS:
            for mode in FORMATS:
                for layout in ["append","leaf"]:
                    name=f"{seed}-{mode}-{layout}";path=OUT/"raw"/(name+".json")
                    if path.exists():
                        case=json.loads(path.read_text())
                        if len(case.get("queries",[]))==4 and case["queries"][0]["scan"].get("protocol"):
                            continue
                        run("examples/disk_ablation_worker.jl",[mode,layout,str(seed),"queries"],name+"-queries")
                        continue
                    run("examples/disk_ablation_worker.jl",[mode,layout,str(seed)],name)
                path=OUT/"quality"/f"{seed}-{mode}.json"
                if not path.exists():run("examples/disk_ablation_quality.jl",[mode,str(seed)],f"quality-{seed}-{mode}")
    for seed in SEEDS:
        for mode in FORMATS:
            path=OUT/"quality"/f"{seed}-{mode}.json"
            if not path.exists():run("examples/disk_ablation_quality.jl",[mode,str(seed)],f"quality-{seed}-{mode}")
    if not args.skip_followups:
        for script,artifact,expected in [
            ("disk_ablation_stress.jl","stress.json",72),
            ("disk_ablation_budget_updates.jl","budget_updates.json",144),
            ("disk_ablation_traffic.jl","traffic.json",144),
            ("disk_ablation_cold_io.jl","cold_io.json",144)]:
            path=OUT/artifact
            if not path.exists() or len(json.loads(path.read_text()))!=expected:
                run("examples/"+script,[],script.removesuffix(".jl"))
        subprocess.run([sys.executable,"scripts/disk_ablation_report.py"],cwd=ROOT,check=True)

if __name__=="__main__": main()
