#!/usr/bin/env bash
# Offline: no model downloads or live source collection. Run after docker build.
set -euo pipefail
image=${1:-mondrian-server:local}
docker info >/dev/null
work=$(mktemp -d)
volume=""
container=""
cleanup() {
    if [ -n "$container" ]; then docker rm -f "$container" >/dev/null 2>&1 || true; fi
    if [ -n "$volume" ]; then docker volume rm "$volume" >/dev/null 2>&1 || true; fi
    rm -rf "$work"
}
trap cleanup EXIT
script_dir=$(cd -- "$(dirname -- "$0")" && pwd)
wait_ready() {
    for _ in $(seq 1 120); do
        if curl -fsS --max-time 5 "$url/readyz" >/dev/null 2>&1; then return; fi
        if [ "$(docker inspect -f '{{.State.Running}}' "$container")" != true ]; then break; fi
        sleep 2
    done
    docker logs "$container"
    return 1
}
for backend in sqlite rocksdb; do
    volume="mondrian-smoke-$backend-$$"
    docker volume create "$volume" >/dev/null
    docker run --rm -v "$volume:/data" "$image" julia --startup-file=no --project=/app/server \
        /app/server/docker/smoke_fixture.jl /data "$backend"
    container=$(docker run -d --read-only --cap-drop ALL --security-opt no-new-privileges \
        --tmpfs /tmp:rw,nosuid,nodev,size=536870912,mode=1777 \
        -v "$volume:/data" -p 127.0.0.1::8080 "$image" --config /data/smoke.toml)
    port=$(docker port "$container" 8080/tcp | head -1 | awk -F: '{print $NF}')
    url="http://127.0.0.1:$port"
    wait_ready
    python3 "$script_dir/check_api.py" "$url" > "$work/before.json"
    # The image's SIGINT stop signal must reach Julia; no SIGKILL or 137 exit.
    docker stop -t 120 "$container" >/dev/null
    test "$(docker inspect -f '{{.State.ExitCode}}' "$container")" = 0
    docker start "$container" >/dev/null
    wait_ready
    python3 "$script_dir/check_api.py" "$url" > "$work/after.json"
    cmp "$work/before.json" "$work/after.json"
    docker stop -t 120 "$container" >/dev/null
    test "$(docker inspect -f '{{.State.ExitCode}}' "$container")" = 0
    docker rm "$container" >/dev/null
    container=""
    docker volume rm "$volume" >/dev/null
    volume=""
    printf '%s container API, graceful stop, persistent identity and restart passed\n' "$backend"
done
