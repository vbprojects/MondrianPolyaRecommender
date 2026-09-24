"""Run from the elastic_embed_colab bundle root: python -m elastic_embed.cli COMMAND --config PATH."""
import argparse
from .config import load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("runtime-check", "prepare-data", "cache-teacher", "initialize", "cache-mohawk", "cache-recovery", "recover", "train", "evaluate", "export", "benchmark"):
        p = sub.add_parser(name)
        p.add_argument("--config", required=True)
        if name in ("cache-teacher", "cache-mohawk", "cache-recovery", "recover", "train"):
            p.add_argument("--resume", action="store_true")
        if name == "runtime-check":
            p.add_argument("--quick", action="store_true", help="Reduced shape matrix for debugging")
        if name == "train":
            p.add_argument("--stage", choices=("mohawk-matrix", "mohawk-block", "full-depth", "prefix", "matryoshka"), required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.command == "runtime-check":
        from .runtime import runtime_check
        runtime_check(cfg, args.quick)
    elif args.command == "prepare-data":
        from .data import prepare
        prepare(cfg)
    elif args.command == "cache-teacher":
        from .teacher import cache_embeddings
        cache_embeddings(cfg, args.resume)
    elif args.command == "initialize":
        from .initialization import initialize
        initialize(cfg)
    elif args.command == "cache-recovery":
        from .teacher import cache_hidden
        cache_hidden(cfg, args.resume)
    elif args.command == "cache-mohawk":
        from .mohawk import cache_alignment
        cache_alignment(cfg, args.resume)
    elif args.command in ("recover", "train"):
        from .training import train
        train(cfg, "recovery" if args.command == "recover" else args.stage, args.resume)
    elif args.command == "evaluate":
        from .evaluation import evaluate
        evaluate(cfg)
    elif args.command == "export":
        from .deployment import export
        export(cfg)
    elif args.command == "benchmark":
        from .deployment import benchmark
        benchmark(cfg)


if __name__ == "__main__":
    main()
