# Packaging validation

Validated on 2026-09-28 in Linux/WSL, Julia 1.13.0. These checks validate the
configuration and application runtime; they are **not a successful Docker build**.

| Check | Result |
| --- | --- |
| Compose base configuration | Passed with Docker Compose 5.1.4 |
| Compose HTTPS profile | Passed configuration parsing |
| Compose attached-disk override | Passed with an explicit data directory |
| Shell syntax and Python helper compilation | Passed |
| Python 3.13 Linux amd64 dependency resolution | Passed: all pinned packages resolve to wheels |
| Pinned encoder, two texts, CPU ONNX execution | Passed: finite normalized 256D outputs |
| Julia worker using MONDRIAN_EMBEDDING_PYTHON | Passed actual CPU inference |
| SQLite local server fixture | 32 posts, 25 unique nearby IDs/deltas, expected keyword counts |
| RocksDB local server fixture | Same checks passed |
| SIGINT shutdown | Both backends exited 0; repeated after restart |
| Restart | Both retained store identity, all 32 posts and keyword counts |
| Actual container build, permissions, health check and restart | Pending: Docker Engine unavailable |
| HTTPS certificate issuance / cloud load | Not performed |

Runtime fixtures were created by `server/docker/smoke_fixture.jl`, served through
`server/bin/serve.jl`, and checked using `server/docker/check_api.py`, before and
after SIGINT and restart. The Docker smoke script repeats these checks using
real containers, a read-only root filesystem and named volumes.

Python wheel resolution used pip with `--python-version 3.13 --implementation cp
--abi cp313 --platform manylinux_2_28_x86_64 --platform manylinux2014_x86_64
--only-binary=:all: --dry-run --ignore-installed`, against the image's requirements.
CPU inference was tested in the workstation's existing Python environment;
resolving Python 3.13 wheels is not the same as running inference in that image.

Neither the WSL Docker command nor the Windows Docker client could reach a
running engine. On a Docker-enabled machine, the remaining verification is:

```sh
docker build --platform linux/amd64 -f server/docker/Dockerfile \
  --target combined -t mondrian-server:local .
bash server/docker/smoke.sh mondrian-server:local
docker build --platform linux/amd64 -f server/docker/Dockerfile \
  --target serve -t mondrian-server:serve .
bash server/docker/smoke.sh mondrian-server:serve
```

Then run the model preparation command in the [deployment guide](README.md) to
verify inference inside the combined image. ARM64 and GPU images are not claimed
or configured by this deployment.
