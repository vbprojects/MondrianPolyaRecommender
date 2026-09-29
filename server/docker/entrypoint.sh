#!/bin/sh
set -eu
if [ "$#" -eq 0 ]; then set -- --help; fi
case "$1" in
    --*)
        exec julia --startup-file=no --project=/app/server /app/server/bin/serve.jl "$@"
        ;;
    *)
        # Maintenance/import/fixture commands use the same image and environment.
        exec "$@"
        ;;
esac
