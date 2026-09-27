from pathlib import Path
import asyncio

import httpx

from labelrobot.api import create_app


def test_http_annotation_and_snapshot(v3_dataset: Path, tmp_path: Path):
    sidecar = tmp_path / "sidecar"
    app = create_app(v3_dataset, sidecar)
    async def exercise():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            dataset = await client.get("/api/dataset")
            assert dataset.status_code == 200
            project = dataset.json()["project"]
            annotation = {
                "id": "segment", "dataset_id": project["dataset_id"], "dataset_version": project["dataset_version"],
                "episode_index": 0, "kind": "subtask", "scope": "segment", "start_frame": 0,
                "end_frame_exclusive": 2, "parent_id": None, "camera_key": None, "label": "grasp", "attributes": {},
            }
            response = await client.post("/api/annotation-commands", json={"command_id": "web", "expected_revision": 0, "operations": [{"action": "upsert", "annotation": annotation}]})
            assert response.json() == {"revision": 1}
            saved = (await client.get("/api/episodes/0/annotations")).json()["annotations"][0]
            assert saved["timestamp_s"] == 0.0
            assert (await client.post("/api/snapshots", json={"snapshot_id": "web"})).json() == {"snapshot_id": "web"}
    try:
        asyncio.run(exercise())
    finally:
        app.state.store.close()
