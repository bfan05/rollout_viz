"""CLI for running the standalone rollout visualization server."""

import argparse
import os
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the standalone rollout accuracy visualization server."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=os.environ.get("ROLLOUT_VIZ_DATA_DIR", "."),
        help="Directory containing rollout JSONL files (1.jsonl, 2.jsonl, ...). Default: current dir or ROLLOUT_VIZ_DATA_DIR.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5000,
        help="Port for the web server (default: 5000).",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind (default: 0.0.0.0).",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.data_dir):
        os.makedirs(args.data_dir, exist_ok=True)

    from ._server import create_app, run_server
    from ._data import load_rollout_data_from_dir

    data_dir = os.path.abspath(args.data_dir)

    def get_data():
        return load_rollout_data_from_dir(data_dir)

    app = create_app(get_data_callback=get_data)
    print(f"Rollout viz server: http://{args.host}:{args.port} (data: {data_dir})")
    sys.stdout.flush()
    run_server(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
