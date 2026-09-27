from __future__ import annotations

import asyncio
import io
import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel, Field

from .readers import DatasetReader, open_dataset
from .storage import AnnotationStore, RevisionConflict, default_sidecar


class CommandRequest(BaseModel):
    command_id: str
    expected_revision: int
    operations: list[dict[str, Any]] = Field(min_length=1)


class SnapshotRequest(BaseModel):
    snapshot_id: str | None = None


def _json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, bytes):
        return "<binary>"
    return value


def create_app(dataset_path: str | Path, annotation_path: str | Path | None = None) -> FastAPI:
    reader = open_dataset(dataset_path)
    sidecar = Path(annotation_path) if annotation_path else default_sidecar(reader.root)
    store = AnnotationStore(sidecar, reader)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        yield
        store.close()

    app = FastAPI(title="LabelRobot", version="0.1.0", lifespan=lifespan)
    app.state.reader = reader
    app.state.store = store
    app.state.resources = {}

    @app.exception_handler(ValueError)
    async def value_error(_: Request, error: ValueError) -> JSONResponse:
        status = 409 if isinstance(error, RevisionConflict) else 422
        return JSONResponse({"detail": str(error)}, status_code=status)

    @app.get("/api/dataset")
    async def dataset() -> dict[str, Any]:
        return {**reader.describe().to_dict(), "project": store.project}

    @app.get("/api/episodes")
    async def episodes(offset: int = 0, limit: int = Query(100, ge=1, le=1000)) -> list[dict[str, Any]]:
        return [episode.__dict__ for episode in reader.list_episodes(offset, limit)]

    @app.get("/api/episodes/{episode_index}/index")
    async def episode_index(episode_index: int) -> dict[str, Any]:
        return reader.get_episode_index(episode_index).__dict__

    @app.get("/api/episodes/{episode_index}/rows")
    async def rows(
        episode_index: int,
        start: int,
        stop: int,
        columns: str = "timestamp,frame_index,observation.state,action",
    ) -> list[dict[str, Any]]:
        requested = [column for column in columns.split(",") if column]
        available = set(reader.describe().features) | {"timestamp", "frame_index", "index", "episode_index", "task_index"}
        selected = [column for column in requested if column in available]
        if not selected:
            raise HTTPException(400, "No valid columns requested")
        return [_json_value(row) for row in reader.read_rows(episode_index, start, stop, selected).to_pylist()]

    @app.get("/api/episodes/{episode_index}/media")
    async def media(episode_index: int) -> list[dict[str, Any]]:
        result = []
        for camera in reader.describe().camera_keys:
            spans = reader.resolve_media(episode_index, camera)
            for span in spans:
                app.state.resources[span.resource_id] = span.path
                item = span.to_dict()
                item.pop("path")
                item["url"] = f"/api/media/{span.resource_id}"
                result.append(item)
        return result

    @app.get("/api/media/{resource_id}")
    async def media_resource(resource_id: str) -> FileResponse:
        path = app.state.resources.get(resource_id)
        if path is None:
            raise HTTPException(404, "Unknown media resource")
        return FileResponse(path)

    @app.get("/api/episodes/{episode_index}/frames/{frame_index}/{camera_key:path}")
    async def exact_frame(episode_index: int, frame_index: int, camera_key: str) -> Response:
        image = await asyncio.to_thread(_decode_frame, reader, episode_index, frame_index, camera_key)
        output = io.BytesIO()
        image.save(output, format="JPEG", quality=90)
        return Response(
            output.getvalue(),
            media_type="image/jpeg",
            headers={"Cache-Control": "private, max-age=31536000, immutable"},
        )

    @app.get("/api/tools")
    async def tools() -> dict[str, Any]:
        return json.loads(store.tools_path.read_text(encoding="utf-8"))

    @app.get("/api/episodes/{episode_index}/annotations")
    async def annotations(episode_index: int) -> dict[str, Any]:
        return {
            "revision": store.revision,
            "annotations": store.list_annotations(episode_index),
            "links": store.list_links(episode_index),
        }

    @app.post("/api/annotation-commands")
    async def annotation_commands(command: CommandRequest) -> dict[str, int]:
        _enrich_operations(command.operations, reader, store)
        revision = store.apply_command(command.command_id, command.expected_revision, command.operations)
        return {"revision": revision}

    @app.post("/api/snapshots")
    async def snapshots(request: SnapshotRequest) -> dict[str, str]:
        return {"snapshot_id": store.create_snapshot(request.snapshot_id)}

    static = Path(__file__).parent / "static"
    if static.exists():
        app.mount("/", StaticFiles(directory=static, html=True), name="ui")
    return app


def _enrich_operations(
    operations: list[dict[str, Any]], reader: DatasetReader, store: AnnotationStore
) -> None:
    project = store.project
    for operation in operations:
        if operation.get("action") != "upsert":
            continue
        annotation = operation["annotation"]
        annotation["dataset_id"] = project["dataset_id"]
        annotation["dataset_version"] = project["dataset_version"]
        if annotation.get("start_frame") is not None:
            annotation["timestamp_s"] = reader.timestamp(
                int(annotation["episode_index"]), int(annotation["start_frame"])
            )
        end = annotation.get("end_frame_exclusive")
        if end is not None:
            length = reader.get_episode_index(int(annotation["episode_index"])).length
            annotation["end_timestamp_s"] = (
                reader.timestamp(int(annotation["episode_index"]), int(end)) if end < length else None
            )


def _decode_frame(reader: DatasetReader, episode_index: int, frame_index: int, camera_key: str) -> Image.Image:
    feature = reader.describe().features.get(camera_key)
    if feature is None or camera_key not in reader.describe().camera_keys:
        raise HTTPException(404, "Unknown camera")
    if feature.get("dtype") == "image":
        value = reader.read_rows(episode_index, frame_index, frame_index + 1, [camera_key])[0][0].as_py()
        if isinstance(value, dict):
            if value.get("bytes"):
                return Image.open(io.BytesIO(value["bytes"])).convert("RGB")
            if value.get("path"):
                path = (reader.root / value["path"]).resolve(strict=True)
                if reader.root not in path.parents:
                    raise HTTPException(403, "Image path escapes dataset")
                return Image.open(path).convert("RGB")
        raise HTTPException(422, "Unsupported image storage")
    spans = reader.resolve_media(episode_index, camera_key)
    if len(spans) != 1:
        raise HTTPException(422, "Unsupported multi-span video")
    timestamp = spans[0].from_timestamp_s + reader.timestamp(episode_index, frame_index)
    try:
        from lerobot.datasets.video_utils import decode_video_frames

        frame = decode_video_frames(spans[0].path, [timestamp], 1e-4, "pyav")[0]
    except Exception as error:
        raise HTTPException(422, f"Video decode failed: {error}") from error
    array = frame.detach().cpu().numpy()
    if array.shape[0] in {1, 3, 4}:
        array = np.moveaxis(array, 0, -1)
    if np.issubdtype(array.dtype, np.floating):
        array = np.clip(array * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(array).convert("RGB")
