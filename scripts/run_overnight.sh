#!/usr/bin/env bash
# Linux/GNU coreutils launcher. Starts a detached, time-limited ingestion process.
set -euo pipefail
set +m # setsid can keep the background child's PID rather than forking again

if [[ ${1:-} == --help || ${1:-} == -h ]]; then
    cat <<'EOF'
Usage: scripts/run_overnight.sh [duration]

Duration defaults to 12h; examples: 8h, 30m, 60s.
Starts in the background, survives terminal closure, and resumes its disk store.

Optional environment variables (relative paths resolve from the repository):
  LIVE_CONFIG         examples/live_sources.toml
  LIVE_STORE          store from LIVE_CONFIG (data/live_disk_gte by default)
  LIVE_LOG_DIR        data/live_runs
  LIVE_JULIA          julia
  LIVE_MAX_EVENTS     1024
  LIVE_SECONDS        10
  LIVE_STORAGE        disk         (disk or arrow)
  TREE_BACKEND        cpu          (cpu or cuda)
  EMBEDDING_DEVICE    cpu          (cpu, cuda, or auto)

The default configuration enables Bluesky. Configure other sources and their
token environment variables in LIVE_CONFIG before launching.
EOF
    exit 0
fi
[[ $# -le 1 ]] || { echo 'Expected only an optional duration; use --help.' >&2; exit 2; }
live_duration=${1:-12h}
[[ $live_duration =~ ^[0-9]+([.][0-9]+)?[smhd]?$ && $live_duration =~ [1-9] ]] || {
    echo 'Duration must be positive, for example 12h or 30m.' >&2; exit 2;
}

live_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd -- "$live_root"
live_config=${LIVE_CONFIG:-examples/live_sources.toml}
live_store=${LIVE_STORE:-}
live_logs=${LIVE_LOG_DIR:-data/live_runs}
live_julia=${LIVE_JULIA:-julia}
live_max_events=${LIVE_MAX_EVENTS:-1024}
live_seconds=${LIVE_SECONDS:-10}
live_backend=${TREE_BACKEND:-cpu}
live_storage=${LIVE_STORAGE:-disk}
live_embedding=${EMBEDDING_DEVICE:-cpu}

for live_program in "$live_julia" timeout nohup setsid flock; do
    command -v -- "$live_program" >/dev/null || { echo "Missing executable: $live_program" >&2; exit 1; }
done
[[ -r $live_config ]] || { echo "Cannot read configuration: $live_config" >&2; exit 1; }
if [[ -z $live_store ]]; then
    live_store=$("$live_julia" --startup-file=no -e 'using TOML; print(get(TOML.parsefile(ARGS[1]),"store","data/live_disk"))' "$live_config")
fi
[[ -x .venv-embeddings/bin/python ]] || { echo 'Prepare .venv-embeddings as described in docs/Live_Ingestion.md first.' >&2; exit 1; }
[[ $live_storage == disk || $live_storage == arrow ]] || { echo 'LIVE_STORAGE must be disk or arrow.' >&2; exit 2; }
[[ $live_storage != disk || $live_backend == cpu ]] || { echo 'Disk tree work uses CPU; EMBEDDING_DEVICE=cuda is supported.' >&2; exit 2; }
[[ $live_backend == cpu || $live_backend == cuda ]] || { echo 'TREE_BACKEND must be cpu or cuda.' >&2; exit 2; }
[[ $live_embedding == cpu || $live_embedding == cuda || $live_embedding == auto ]] || {
    echo 'EMBEDDING_DEVICE must be cpu, cuda, or auto.' >&2; exit 2;
}
[[ $live_max_events =~ ^[1-9][0-9]{0,3}$ ]] && ((live_max_events <= 1024)) || {
    echo 'LIVE_MAX_EVENTS must be between 1 and 1024.' >&2; exit 2;
}
[[ $live_seconds =~ ^[0-9]+([.][0-9]+)?$ && $live_seconds =~ [1-9] ]] || {
    echo 'LIVE_SECONDS must be positive.' >&2; exit 2;
}

mkdir -p -- "$live_store" "$live_logs"
# This is an early check; the Julia writer takes the authoritative lifetime lock.
flock -n "$live_store/.writer.lock" true || { echo "An ingestion writer already owns $live_store." >&2; exit 1; }
live_logs=$(cd -- "$live_logs" && pwd)
if [[ -e $live_logs/latest && ! -L $live_logs/latest ]]; then
    echo "$live_logs/latest already exists and is not a symlink." >&2; exit 1
fi
live_run=$(mktemp -d "$live_logs/$(date -u +%Y%m%dT%H%M%SZ)-XXXXXX")
live_log=$live_run/ingest.log
live_command=("$live_julia" --project=environments/ingestion examples/ingest_live.jl
    --storage "$live_storage" --config "$live_config" --store "$live_store" --loop
    --max-events "$live_max_events" --seconds "$live_seconds"
    --backend "$live_backend" --embedding-device "$live_embedding")

{
    printf 'Started UTC: %s\nDuration: %s\nStore: %s\n' "$(date -u +%FT%TZ)" "$live_duration" "$live_store"
    printf 'Command:'; printf ' %q' "${live_command[@]}"; printf '\n'
} > "$live_log"
# Keep timeout's default process-group handling: it stops Julia AND its embedding
# worker. TERM can interrupt an uncommitted batch; the durable SQLite generation
# (or legacy Arrow CURRENT) retains its matching cursor for replay.
nohup setsid timeout --verbose --signal=TERM --kill-after=60s "$live_duration" \
    "${live_command[@]}" >> "$live_log" 2>&1 < /dev/null &
live_pid=$!
printf '%s\n' "$live_pid" > "$live_run/runner.pid"
ln -sfnT -- "$live_run" "$live_logs/latest"

printf 'Launched for %s (timeout PID %s).\n' "$live_duration" "$live_pid"
printf 'Log: %s\n' "$live_log"
printf 'Monitor: tail -f %q\n' "$live_log"
printf 'Stop early: kill -TERM %s\n' "$live_pid"
