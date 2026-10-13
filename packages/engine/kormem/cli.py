"""CLI entry for kormem-engine."""

from __future__ import annotations

import argparse

from kormem.core.config import ensureKormemLayout, loadConfig
from kormem.server.app import runServer


def main() -> None:
    parser = argparse.ArgumentParser(prog="kormem-engine")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("init", help="Initialize ~/.kormem layout")
    serve = sub.add_parser("serve", help="Start HTTP API server")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=None)

    args = parser.parse_args()
    if args.command == "init":
        home = ensureKormemLayout()
        print(f"Initialized kormem at {home}")
    elif args.command == "serve":
        runServer(host=args.host, port=args.port)
    else:
        config = loadConfig()
        runServer(port=config.enginePort)


if __name__ == "__main__":
    main()
