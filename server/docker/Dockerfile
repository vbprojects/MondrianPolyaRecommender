# syntax=docker/dockerfile:1
ARG JULIA_IMAGE=julia:1.13.0-trixie
FROM $JULIA_IMAGE AS base
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl tini libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 10001 mondrian \
    && useradd --uid 10001 --gid 10001 --create-home mondrian \
    && mkdir -p /data /home/mondrian/.cache/huggingface \
    && chown -R 10001:10001 /data /home/mondrian
ENV JULIA_CPU_TARGET=generic \
    JULIA_NUM_THREADS=2,1 \
    OPENBLAS_NUM_THREADS=1 \
    OMP_NUM_THREADS=2 \
    JULIA_DEPOT_PATH=/tmp/julia-depot:/opt/julia-depot \
    HF_HOME=/home/mondrian/.cache/huggingface \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1
WORKDIR /app

FROM base AS julia-deps
COPY server/Project.toml server/Manifest.toml /app/server/
RUN JULIA_DEPOT_PATH=/opt/julia-depot JULIA_NUM_PRECOMPILE_TASKS=2 \
    julia --startup-file=no --project=/app/server \
    -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'

# No Python or encoder required when serving an existing store.
FROM base AS serve
COPY --from=julia-deps /opt/julia-depot /opt/julia-depot
COPY server/Project.toml server/Manifest.toml /app/server/
COPY src/ /app/src/
COPY server/src/ /app/server/src/
COPY server/bin/ /app/server/bin/
COPY server/inspector/ /app/server/inspector/
COPY server/docker/ /app/server/docker/
COPY julia-yake/src/ /app/julia-yake/src/
COPY julia-yake/data/ /app/julia-yake/data/
COPY julia-yake/notices/ /app/julia-yake/notices/
RUN chmod +x /app/server/docker/entrypoint.sh \
    && julia --startup-file=no --project=/app/server /app/server/bin/serve.jl --help \
    && rm -rf /tmp/julia-depot
USER 10001:10001
EXPOSE 8080
# Julia receives SIGINT so its finally blocks drain work and close the store.
STOPSIGNAL SIGINT
HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=3 \
    CMD curl --fail --silent --show-error --max-time 8 http://127.0.0.1:8080/readyz || exit 1
ENTRYPOINT ["/usr/bin/tini", "--", "/app/server/docker/entrypoint.sh"]
CMD ["--config", "/app/server/docker/serve.toml"]

# Default image: CPU embedding worker plus native Julia YAKE, ingestion and API.
FROM serve AS combined
USER root
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-venv \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m venv /opt/embeddings
COPY scripts/requirements-embeddings.txt /app/scripts/requirements-embeddings.txt
COPY server/docker/requirements.txt /app/server/docker/requirements.txt
RUN /opt/embeddings/bin/pip install --no-cache-dir -r /app/server/docker/requirements.txt \
    && /opt/embeddings/bin/pip check
COPY scripts/live_embedding_worker.py scripts/embed_bluesky.py /app/scripts/
ENV MONDRIAN_EMBEDDING_PYTHON=/opt/embeddings/bin/python
USER 10001:10001
CMD ["--config", "/app/server/docker/combined.toml"]
