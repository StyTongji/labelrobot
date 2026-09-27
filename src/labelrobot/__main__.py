from __future__ import annotations

import argparse
import threading
import webbrowser
from pathlib import Path

from .readers import open_dataset
from .storage import AnnotationStore, default_sidecar


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(prog="labelrobot")
    result.add_argument("command_or_dataset")
    result.add_argument("dataset", nargs="?")
    result.add_argument("--annotations", type=Path)
    result.add_argument("--host", default="127.0.0.1")
    result.add_argument("--port", type=int, default=8765)
    result.add_argument("--no-browser", action="store_true")
    return result


def main() -> None:
    arguments = parser().parse_args()
    if arguments.command_or_dataset == "snapshot":
        if arguments.dataset is None:
            parser().error("snapshot requires DATASET")
        reader = open_dataset(arguments.dataset)
        sidecar = arguments.annotations or default_sidecar(reader.root)
        store = AnnotationStore(sidecar, reader)
        try:
            print(store.create_snapshot())
        finally:
            store.close()
        return
    if arguments.dataset is not None:
        parser().error("unexpected second dataset argument")
    try:
        import uvicorn
    except ImportError as error:
        raise SystemExit("Install LabelRobot web dependencies with: pip install -e .") from error
    from .api import create_app

    url = f"http://{arguments.host}:{arguments.port}"
    if not arguments.no_browser:
        threading.Timer(0.8, webbrowser.open, args=(url,)).start()
    uvicorn.run(
        create_app(arguments.command_or_dataset, arguments.annotations),
        host=arguments.host,
        port=arguments.port,
    )


if __name__ == "__main__":
    main()
