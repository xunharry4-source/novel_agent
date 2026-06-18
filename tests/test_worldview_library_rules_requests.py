import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from real_request_test_utils import cleanup_world, create_world, get_worldview_for_world, list_lore, request_json, unique_suffix


def test_world_create_auto_provisions_unique_worldview_library():
    suffix = unique_suffix("worldview_library_auto")
    world_id = f"world_{suffix}"
    try:
        created_world = create_world(world_id=world_id, name=f"World {suffix}", summary="创建世界时自动生成唯一 worldview 库")
        worldview = get_worldview_for_world(world_id)
        assert worldview["world_id"] == world_id, worldview
        assert worldview["worldview_id"] == created_world["worldview_id"], {"created_world": created_world, "worldview": worldview}
        assert worldview.get("auto_created") is True, worldview
        assert worldview["name"], worldview
    finally:
        cleanup_world(world_id)


def test_worldview_create_api_is_disabled_after_world_auto_provision():
    suffix = unique_suffix("worldview_library_disabled_create")
    world_id = f"world_{suffix}"
    try:
        create_world(world_id=world_id, name=f"World {suffix}", summary="禁用 create worldview")
        payload = request_json(
            "POST",
            "/api/worldviews/create",
            expected_status=409,
            json={"world_id": world_id, "name": "Should Fail", "summary": "此接口已废弃"},
        )
        assert payload["status"] == "error", payload
        assert "Create worldview is disabled" in payload["error"], payload
    finally:
        cleanup_world(world_id)


def test_delete_world_auto_removes_worldview_library_and_its_settings():
    suffix = unique_suffix("worldview_library_delete")
    world_id = f"world_{suffix}"
    try:
        create_world(world_id=world_id, name=f"World {suffix}", summary="删除世界自动删除 worldview 库")
        worldview = get_worldview_for_world(world_id)
        request_json(
            "POST",
            "/api/archive/update",
            json={
                "id": f"wv_entry_{suffix}",
                "type": "worldview",
                "world_id": world_id,
                "worldview_id": worldview["worldview_id"],
                "name": f"Rule {suffix}",
                "content": "自动清理验证条目",
                "category": "世界观设定",
            },
        )
        entries = list_lore(world_id=world_id, worldview_id=worldview["worldview_id"], page=1, page_size=20)
        assert any(item["id"] == f"wv_entry_{suffix}" for item in entries), entries

        request_json("DELETE", "/api/worlds/delete", json={"world_id": world_id, "cascade": True})

        worldviews_after = request_json("GET", "/api/worldviews/list", params={"world_id": world_id, "page": 1, "page_size": 20})
        assert worldviews_after == [], worldviews_after
        lore_after = request_json(
            "GET",
            "/api/lore/list",
            params={"world_id": world_id, "worldview_id": worldview["worldview_id"], "page": 1, "page_size": 20},
        )
        assert lore_after == [], lore_after
    finally:
        try:
            cleanup_world(world_id)
        except AssertionError:
            pass


def test_worldviews_list_self_heals_missing_unique_library():
    suffix = unique_suffix("worldview_library_self_heal")
    world_id = f"world_{suffix}"
    try:
        created_world = create_world(world_id=world_id, name=f"World {suffix}", summary="列表接口应自动补回缺失 worldview 库")
        worldview = get_worldview_for_world(world_id)
        request_json("DELETE", "/api/worldviews/delete", json={"worldview_id": worldview["worldview_id"], "cascade": True})

        repaired = request_json("GET", "/api/worldviews/list", params={"world_id": world_id, "page": 1, "page_size": 20})
        assert isinstance(repaired, list), repaired
        assert len(repaired) == 1, repaired
        assert repaired[0]["world_id"] == world_id, repaired
        assert repaired[0]["worldview_id"] == created_world["worldview_id"], {"created_world": created_world, "repaired": repaired}
        assert repaired[0].get("auto_created") is True, repaired
    finally:
        cleanup_world(world_id)
