# Docker deployment

Run one server/ingester on a Linux x86-64 VM with persistent local storage.
Defaults: CPU inference, Float16, hashtag summaries, seven-day retention and a
six-hour density half-life. New stores use SQLite; RocksDB remains selectable.
Existing stores are detected without conversion. New combined deployments use
pinned 384D GTE-tiny float ONNX inference. A Qwen store cannot resume under this
configuration: use a fresh volume/store or retain an explicit Qwen source config.

Use a **Lightsail Linux instance**, or an equivalent VM, with Docker Compose and
a persistent SSD/block volume. This package requires filesystem locking and
fsync; it is not a stateless Lightsail Container Services deployment.
Attached Lightsail disks can be backed up independently of the instance
([AWS storage guidance](https://docs.aws.amazon.com/lightsail/latest/userguide/amazon-lightsail-faq-block-storage.html)).

## Build and test

From the repository root, with Docker Engine and Compose installed:

```sh
docker build --platform linux/amd64 -f server/docker/Dockerfile \
  --target combined -t mondrian-server:local .
bash server/docker/smoke.sh mondrian-server:local
```

The smoke test creates private SQLite and RocksDB fixtures, exercises all three
APIs and readiness, stops with SIGINT, restarts, and checks persistent identity,
post counts and keywords. It removes only its own temporary containers/volumes.
It does not ingest live posts or download an encoder.

The image uses Julia 1.13.0 and the committed server Manifest. Generic CPU code
avoids baking workstation-specific instructions into the Julia cache. Trixie
provides Python 3.13; pinned NumPy requires Python 3.12+. CPU Python dependencies
and coincurve are pinned. Model weights use their existing pinned revision.
Base images use release tags; record the built image digest for deployment.

The build context excludes corpora, live databases, model caches, notebooks,
virtual environments and credentials. The image runs as UID/GID **10001**,
with a read-only root filesystem under Compose.

**Validation:** see [VALIDATION.md](VALIDATION.md). Docker Engine was unavailable
on the packaging workstation; run the actual image build and container smoke
test above before deployment.

## Start a pilot

```sh
cd deploy
cp .env.example .env
docker compose config --quiet
docker compose build
# Download the encoder into the persistent cache and verify real CPU inference.
docker compose run --rm --no-deps mondrian \
  /opt/embeddings/bin/python /app/server/docker/prepare_model.py
docker compose up -d
docker compose ps
docker compose logs --tail=100 -f mondrian
```

Edit `config/server.toml` and `config/sources.toml` first if needed. Container
paths are `/data/store` and `/config/*.toml`. Keep source names stable: they own
persisted cursors. Optional source tokens go in the ignored `.env`, not an image
or TOML file. Public Bluesky ingestion needs no token.

The server binds all interfaces inside the container; Compose exposes it only
on host loopback at port 8080:

```sh
curl --fail http://127.0.0.1:8080/readyz
curl --fail http://127.0.0.1:8080/v1/tree
# To inspect a VM without publishing its API:
ssh -L 8080:127.0.0.1:8080 ubuntu@YOUR_VM_IP
```

Readiness checks storage and background task status, not encoder inference.
The preparation command verifies inference separately. Compilation, backfill
or recovery can delay readiness; increase the three-minute health-check start
period for larger stores if necessary.

## Lightsail or another VM

Start a pilot with **2 vCPUs and 4 GiB RAM**. This is an initial allocation, not a
measured cloud capacity guarantee. Compose caps the app at 3 GiB and two CPUs,
leaving room for the OS and proxy. Increase VM RAM and `MONDRIAN_MEMORY` together
if RSS or OOM events require it. The 256 MiB managed storage cache and 32 MiB
server allowance are only part of memory use; inference, Julia/native libraries,
compilation and the writable `/tmp` cache also consume memory.

1. Create an x86-64 Linux VM; use a static public IP for a domain.
2. Install [Docker Engine and Compose](https://docs.docker.com/engine/install/ubuntu/).
3. Attach and persistently mount a disk at `/srv/mondrian`, following
   [AWS's disk instructions](https://docs.aws.amazon.com/lightsail/latest/userguide/create-and-attach-additional-block-storage-disks-linux-unix.html).
4. Create the application directory:

   ```sh
   sudo install -d -o 10001 -g 10001 -m 750 /srv/mondrian/data
   ```

5. Set `MONDRIAN_DATA_DIR=/srv/mondrian/data` in `deploy/.env` and use both files
   for **all** subsequent Compose commands:

   ```sh
   docker compose -f compose.yaml -f compose.disk.yaml config --quiet
   docker compose -f compose.yaml -f compose.disk.yaml up -d
   ```

The override refuses to create a missing host directory. Verify the data disk is
mounted before deployment and after reboot; a directory alone does not prove the
expected filesystem is mounted. Ensure Docker starts after this mount is
available. Default named volumes otherwise live on Docker's own disk.

At 100 posts/s, seven days contain 60.48 million posts. Float16 384D vectors
alone would occupy **46.45 GB decimal**, before text, metadata, keywords, PPCA,
indexes, logs and reclamation space. Size storage from observed total bytes/post
with compaction headroom; this pilot does not validate weekly capacity. Build
on a workstation/CI host if building would exhaust the VM's memory or disk.

## HTTPS

Point domain DNS at the VM; set `DOMAIN` in `.env`. Allow inbound TCP 80 and 443
in the cloud firewall, keeping application port 8080 private. Then:

```sh
docker compose --profile https up -d
# With the attached data disk:
docker compose -f compose.yaml -f compose.disk.yaml --profile https up -d
```

Caddy terminates HTTPS and forwards over the private container network.
Certificate state persists in named volumes. Add exact browser origins to
`http.cors_origins`. CORS is not authentication: this is a public read API with
existing request/body/queue limits. Monitor public load and add upstream quotas
if needed. Run **one replica per store**; even serve-only mode performs
maintenance and owns the writer lock.

## Existing stores and serve-only images

Copy a **closed, consistent full store directory** into `/data/store` on the
volume, retaining metadata, payloads and subdirectories. Set ownership of the
copied directory to 10001:10001. A SQLite file alone is not a tree backup.
Do not copy a live store without coordinating its writer and payload publication.

For serving without Python/inference:

```sh
docker build --platform linux/amd64 -f server/docker/Dockerfile \
  --target serve -t mondrian-server:serve .
```

Set `MONDRIAN_IMAGE=mondrian-server:serve`, `MONDRIAN_TARGET=serve` and
`mode = "serve"` in `config/server.toml`. Skip model preparation. An existing
store is required; expiry and maintenance remain enabled. Historical fixtures
expire using the current wall clock; inspect those with the benchmark harness.

For a new RocksDB store, set `storage.backend = "rocksdb"` before first start.
For an existing SQLite store, stop the owner and use the explicit
[import tool](../server/RocksDB.md). Changing a setting does not migrate data.
Old RocksDB rows remain readable, but the first write upgrades the format marker;
older binaries then reject it.

## Updates, backups and recovery

Use a versioned image tag or immutable registry digest in `MONDRIAN_IMAGE`.
Without a registry, transfer a built image:

```sh
docker save mondrian-server:local | gzip > mondrian-server.tar.gz
# Transfer to the VM, then:
gunzip -c mondrian-server.tar.gz | docker load
```

For registry images, use `docker compose pull mondrian` followed by
`docker compose up -d --no-build`. Publishing and cloud provisioning are separate
deployment actions.

Compose sends SIGINT and allows two minutes to drain work and close the store.
Forced termination uses existing transaction recovery on reopen. Increase
`stop_grace_period` for long maintenance jobs. `unless-stopped` handles process
exits and Docker restarts; an `unhealthy` status alone does not restart a container.

For an offline backup, stop the owner and archive `/data` or snapshot the data
disk. From `deploy/`, using the same Compose files as your deployment:

```sh
docker compose stop mondrian
docker compose run --rm --no-deps mondrian tar -C /data -czf - store \
  > mondrian-store-backup.tar.gz
docker compose up -d mondrian
```

Check that the archive command succeeded. Keep the previous image, Manifest,
config and pre-upgrade store backup together for rollback. Test restoring to
a separate volume before replacing the active store. Do not use
`docker compose down -v` on data you intend to retain.

Monitor readiness, `docker stats`, free disk space, ingestion errors, cursor
progress and query latency. Logs rotate at 10 MB × 3 files per service. No GPU
is required; cloud throughput and sustained mixed-query capacity remain unmeasured.
