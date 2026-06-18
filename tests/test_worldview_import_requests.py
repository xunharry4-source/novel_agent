import os
import tempfile
import threading
import time
import uuid
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

API_BASE_URL = os.environ.get("API_BASE_URL", "http://127.0.0.1:5017")


def start_real_api_if_requested():
    if os.environ.get("START_TEST_SERVER") != "1":
        return None

    from werkzeug.serving import make_server
    from app_api import app

    host = "127.0.0.1"
    port = int(API_BASE_URL.rsplit(":", 1)[1])
    server = make_server(host, port, app)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    for _ in range(50):
        try:
            response = requests.get(f"{API_BASE_URL}/api/system/health", timeout=2)
            if response.status_code < 500:
                return server
        except Exception:
            time.sleep(0.1)
    raise RuntimeError("Real API server did not start")


def request_json(method, path, **kwargs):
    response = requests.request(method, f"{API_BASE_URL}{path}", timeout=20, **kwargs)
    try:
        payload = response.json()
    except Exception:
        payload = response.text
    assert response.ok, f"{method} {path} failed: {response.status_code} {payload}"
    return payload


def upload_and_verify(world_id, worldview_id, file_path, expected_name, expected_path):
    with open(file_path, "rb") as handle:
        payload = request_json(
            "POST",
            "/api/worldviews/import",
            data={"world_id": world_id, "worldview_id": worldview_id},
            files={"file": (file_path.name, handle)},
        )

    assert payload["status"] == "success"
    assert payload["world_id"] == world_id
    assert payload["worldview_id"] == worldview_id
    assert payload["imported_count"] >= 1
    assert any(item["path"] == expected_path for item in payload["entries"]), payload

    listed = request_json(
        "GET",
        "/api/lore/list",
        params={
            "world_id": world_id,
            "worldview_id": worldview_id,
            "query": expected_name,
            "page": 1,
            "page_size": 50,
        },
    )
    matched = [item for item in listed if item["name"] == expected_name and item.get("category") == expected_path]
    assert matched, listed
    assert all(item.get("world_id") == world_id for item in matched)
    assert all(item.get("worldview_id") == worldview_id for item in matched)


def main():
    server = start_real_api_if_requested()
    test_worldview_id = None
    test_world_id = None
    try:
        marker = uuid.uuid4().hex[:8]
        created_world = request_json(
            "POST",
            "/api/worlds/create",
            json={
                "world_id": f"world_import_test_{marker}",
                "name": f"导入测试世界 {marker}",
                "summary": "world create should auto-provision unique worldview library",
            },
        )
        test_world_id = created_world["world_id"]
        worldviews = request_json(
            "GET",
            "/api/worldviews/list",
            params={"world_id": test_world_id, "page": 1, "page_size": 10},
        )
        assert len(worldviews) == 1, worldviews
        test_worldview_id = worldviews[0]["worldview_id"]
        request_json(
            "POST",
            "/api/worldviews/update",
            json={
                "worldview_id": test_worldview_id,
                "name": f"导入测试世界观库 {marker}",
                "summary": "用于真实 requests 导入测试",
            },
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            json_file = base / "hierarchy.json"
            json_file.write_text(
                '{"name":"文明体系","children":[{"name":"政治","children":[{"name":"议会","content":"三院制"}]}]}',
                encoding="utf-8",
            )
            md_file = base / "hierarchy.md"
            md_file.write_text("# 地理\n\n## 北境\n\n寒冷边疆。\n", encoding="utf-8")
            opml_file = base / "hierarchy.opml"
            opml_file.write_text(
                '<?xml version="1.0" encoding="UTF-8"?><opml version="2.0"><body><outline text="魔法"><outline text="符文" _note="符文体系"/></outline></body></opml>',
                encoding="utf-8",
            )
            opml_structured_file = base / "hierarchy_structured.opml"
            opml_structured_file.write_text(
                '<?xml version="1.0" encoding="UTF-8"?><opml version="1.0"><body><outline text="种族"><outline text="碳基生命"><outline text="秦族"><outline text="最高标志"/><outline text="拥有独立文明"/></outline></outline></outline></body></opml>',
                encoding="utf-8",
            )
            xml_file = base / "hierarchy.xml"
            xml_file.write_text(
                '<root><technology><propulsion><warp_drive>曲率航行依赖稳定场</warp_drive></propulsion></technology></root>',
                encoding="utf-8",
            )

            upload_and_verify(test_world_id, test_worldview_id, json_file, "议会", "文明体系 > 政治 > 议会")
            upload_and_verify(test_world_id, test_worldview_id, md_file, "北境", "地理 > 北境")
            upload_and_verify(test_world_id, test_worldview_id, opml_file, "符文", "魔法 > 符文")
            upload_and_verify(test_world_id, test_worldview_id, opml_structured_file, "秦族", "种族 > 碳基生命 > 秦族")
            upload_and_verify(test_world_id, test_worldview_id, xml_file, "warp_drive", "root > technology > propulsion > warp_drive")

        print({
            "status": "passed",
            "api_base_url": API_BASE_URL,
            "world_id": test_world_id,
            "worldview_id": test_worldview_id,
            "formats": ["json", "md", "xml", "opml"],
        })
    finally:
        if test_world_id:
            response = requests.delete(
                f"{API_BASE_URL}/api/worlds/delete",
                json={"world_id": test_world_id, "cascade": True},
                timeout=20,
            )
            assert response.ok, f"cleanup failed: {response.status_code} {response.text}"
        if server:
            server.shutdown()


if __name__ == "__main__":
    main()
