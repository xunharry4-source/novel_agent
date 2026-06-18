import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from real_request_test_utils import cleanup_world, create_world, create_worldview, get_worldview, request_json, unique_suffix


def test_real_worldview_policy_update_requests():
    suffix = unique_suffix("worldview_policy_update")
    world_id = f"world_{suffix}"
    updated_rules = ["禁止世界观条目与灯塔公会登记制度冲突"]
    updated_settings = {"canon_scope": "北港灯塔", "continuity_rule": "所有潮汐刻度必须可追溯"}
    try:
        world = create_world(world_id=world_id, name=f"World {suffix}", summary="世界观规则字段更新父级世界")
        worldview = create_worldview(
            world_id=world_id,
            worldview_id=world["worldview_id"],
            name=f"Worldview {suffix}",
            summary="原始世界观",
        )
        worldview_id = worldview["worldview_id"]
        response = request_json(
            "POST",
            "/api/worldviews/update",
            json={"worldview_id": worldview_id, "name": f"Worldview Updated {suffix}", "summary": "已更新世界观", "forbidden_rules": updated_rules, "basic_settings": updated_settings},
        )
        assert response.get("status") == "success", response
        queried = get_worldview(worldview_id)
        assert queried is not None, worldview_id
        assert queried.get("name") == f"Worldview Updated {suffix}", queried
        assert queried.get("forbidden_rules") == updated_rules, queried
        assert queried.get("basic_settings") == updated_settings, queried
    finally:
        cleanup_world(world_id)
