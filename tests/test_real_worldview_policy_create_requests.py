import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from real_request_test_utils import cleanup_world, create_world, get_worldview_for_world, unique_suffix


def test_real_worldview_policy_create_requests():
    suffix = unique_suffix("worldview_policy_create")
    world_id = f"world_{suffix}"
    try:
        created_world = create_world(world_id=world_id, name=f"World {suffix}", summary="世界观规则字段父级世界")
        queried = get_worldview_for_world(world_id)
        assert queried.get("world_id") == world_id, queried
        assert queried.get("worldview_id") == created_world.get("worldview_id"), queried
        assert queried.get("auto_created") is True, queried
        assert queried.get("name"), queried
        assert queried.get("summary") == "世界观规则字段父级世界", queried
        assert queried.get("forbidden_rules") == [], queried
        assert queried.get("basic_settings") == {}, queried
    finally:
        cleanup_world(world_id)
