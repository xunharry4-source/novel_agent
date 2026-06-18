import os
import uuid

import requests


BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:5006")
API_PREFIX = f"{BASE_URL}/api"
TIMEOUT = 10


def assert_json_response(response: requests.Response, expected_status: int = 200):
    if response.status_code != expected_status:
        raise AssertionError(
            f"HTTP {response.status_code} != {expected_status}\n"
            f"URL: {response.request.method} {response.url}\n"
            f"Body: {response.text}"
        )
    try:
        return response.json()
    except Exception as exc:
        raise AssertionError(f"Response is not JSON: {response.text}") from exc


def get_world(world_id: str):
    data = assert_json_response(requests.get(f"{API_PREFIX}/worlds/list", timeout=TIMEOUT))
    assert isinstance(data, list), data
    return next((item for item in data if item.get("world_id") == world_id), None)


def get_worldview(worldview_id: str):
    data = assert_json_response(requests.get(f"{API_PREFIX}/worldviews/list", params={"worldview_id": worldview_id, "page": 1, "page_size": 10}, timeout=TIMEOUT))
    assert isinstance(data, list), data
    return next((item for item in data if item.get("worldview_id") == worldview_id), None)


def get_novel(novel_id: str):
    data = assert_json_response(requests.get(f"{API_PREFIX}/novels/list", params={"novel_id": novel_id, "page": 1, "page_size": 10}, timeout=TIMEOUT))
    assert isinstance(data, list), data
    return next((item for item in data if item.get("novel_id") == novel_id), None)


def get_outline(outline_id: str):
    data = assert_json_response(requests.get(f"{API_PREFIX}/outlines/list", params={"outline_id": outline_id, "page": 1, "page_size": 10}, timeout=TIMEOUT))
    assert isinstance(data, list), data
    return next((item for item in data if item.get("outline_id") == outline_id), None)


def get_chapter(chapter_id: str, outline_id: str, worldview_id: str):
    data = assert_json_response(
        requests.get(
            f"{API_PREFIX}/lore/list",
            params={"outline_id": outline_id, "worldview_id": worldview_id, "page": 1, "page_size": 20},
            timeout=TIMEOUT,
        )
    )
    assert isinstance(data, list), data
    return next((item for item in data if item.get("id") == chapter_id), None)


def create_full_chain(suffix: str):
    world_name = f"Strict World {suffix}"
    world_summary = f"Strict World Summary {suffix}"
    world = assert_json_response(
        requests.post(f"{API_PREFIX}/worlds/create", json={"name": world_name, "summary": world_summary}, timeout=TIMEOUT)
    )
    assert world.get("status") == "success", world
    world_id = world["world_id"]
    queried_world = get_world(world_id)
    assert queried_world["name"] == world_name, queried_world
    assert queried_world["summary"] == world_summary, queried_world

    worldview_name = f"Strict Worldview {suffix}"
    worldview_seed = assert_json_response(
        requests.get(
            f"{API_PREFIX}/worldviews/list",
            params={"world_id": world_id, "page": 1, "page_size": 10},
            timeout=TIMEOUT,
        )
    )
    assert len(worldview_seed) == 1, worldview_seed
    worldview_id = worldview_seed[0]["worldview_id"]
    worldview = assert_json_response(
        requests.post(
            f"{API_PREFIX}/worldviews/update",
            json={"worldview_id": worldview_id, "name": worldview_name, "summary": f"WV Summary {suffix}"},
            timeout=TIMEOUT,
        )
    )
    assert worldview.get("status") == "success", worldview
    queried_worldview = get_worldview(worldview_id)
    assert queried_worldview["name"] == worldview_name, queried_worldview
    assert queried_worldview["world_id"] == world_id, queried_worldview

    novel_name = f"Strict Novel {suffix}"
    novel = assert_json_response(
        requests.post(
            f"{API_PREFIX}/novels/create",
            json={"name": novel_name, "summary": f"Novel Summary {suffix}", "world_id": world_id},
            timeout=TIMEOUT,
        )
    )
    assert novel.get("status") == "success", novel
    novel_id = novel["novel_id"]
    queried_novel = get_novel(novel_id)
    assert queried_novel["name"] == novel_name, queried_novel
    assert queried_novel["world_id"] == world_id, queried_novel
    assert "worldview_id" not in queried_novel, queried_novel

    outline_name = f"Strict Outline {suffix}"
    outline = assert_json_response(
        requests.post(
            f"{API_PREFIX}/outlines/create",
            json={"name": outline_name, "summary": f"Outline Summary {suffix}", "novel_id": novel_id, "worldview_id": worldview_id},
            timeout=TIMEOUT,
        )
    )
    assert outline.get("status") == "success", outline
    outline_id = outline["outline_id"]
    queried_outline = get_outline(outline_id)
    assert queried_outline["novel_id"] == novel_id, queried_outline
    assert queried_outline["worldview_id"] == worldview_id, queried_outline
    assert queried_outline["world_id"] == world_id, queried_outline

    chapter_id = f"chapter_{suffix}"
    chapter_title = f"Strict Chapter {suffix}"
    chapter_content = f"Strict Chapter Content {suffix}"
    chapter = assert_json_response(
        requests.post(
            f"{API_PREFIX}/archive/update",
            json={
                "id": chapter_id,
                "type": "prose",
                "name": chapter_title,
                "content": chapter_content,
                "outline_id": outline_id,
                "novel_id": novel_id,
                "worldview_id": worldview_id,
                "world_id": world_id,
            },
            timeout=TIMEOUT,
        )
    )
    assert chapter.get("status") == "success", chapter
    queried_chapter = get_chapter(chapter_id, outline_id, worldview_id)
    assert queried_chapter["name"] == chapter_title, queried_chapter
    assert queried_chapter["content"] == chapter_content, queried_chapter
    assert queried_chapter["novel_id"] == novel_id, queried_chapter
    assert queried_chapter["world_id"] == world_id, queried_chapter

    return {
        "world_id": world_id,
        "worldview_id": worldview_id,
        "novel_id": novel_id,
        "outline_id": outline_id,
        "chapter_id": chapter_id,
    }


def assert_tree_contains(chain: dict):
    tree = assert_json_response(
        requests.get(f"{API_PREFIX}/world-hierarchy/tree", params={"world_id": chain["world_id"], "page": 1, "page_size": 50}, timeout=TIMEOUT)
    )
    assert tree.get("status") == "success", tree
    world = next((item for item in tree["worlds"] if item["world_id"] == chain["world_id"]), None)
    assert world is not None, tree
    worldview = next((item for item in world["worldviews"] if item["worldview_id"] == chain["worldview_id"]), None)
    assert worldview is not None, world
    novel = next((item for item in world["novels"] if item["novel_id"] == chain["novel_id"]), None)
    assert novel is not None, world
    outline = next((item for item in novel["outlines"] if item["outline_id"] == chain["outline_id"]), None)
    assert outline is not None, novel
    chapter = next((item for item in outline["chapters"] if item["id"] == chain["chapter_id"]), None)
    assert chapter is not None, outline


def test_world_hierarchy_lifecycle():
    suffix = uuid.uuid4().hex[:10]
    chain = {}
    try:
        chain = create_full_chain(suffix)
        assert_tree_contains(chain)

        updated_world_name = f"Strict World Updated {suffix}"
        assert_json_response(
            requests.post(
                f"{API_PREFIX}/worlds/update",
                json={"world_id": chain["world_id"], "name": updated_world_name, "summary": f"World Updated Summary {suffix}"},
                timeout=TIMEOUT,
            )
        )
        queried_world = get_world(chain["world_id"])
        assert queried_world["name"] == updated_world_name, queried_world
        assert queried_world["summary"] == f"World Updated Summary {suffix}", queried_world

        updated_worldview_name = f"Strict Worldview Updated {suffix}"
        assert_json_response(
            requests.post(
                f"{API_PREFIX}/worldviews/update",
                json={"worldview_id": chain["worldview_id"], "name": updated_worldview_name, "summary": f"WV Updated {suffix}"},
                timeout=TIMEOUT,
            )
        )
        queried_worldview = get_worldview(chain["worldview_id"])
        assert queried_worldview["name"] == updated_worldview_name, queried_worldview
        assert queried_worldview["summary"] == f"WV Updated {suffix}", queried_worldview

        updated_novel_name = f"Strict Novel Updated {suffix}"
        assert_json_response(
            requests.post(
                f"{API_PREFIX}/novels/update",
                json={"novel_id": chain["novel_id"], "name": updated_novel_name, "summary": f"Novel Updated {suffix}"},
                timeout=TIMEOUT,
            )
        )
        queried_novel = get_novel(chain["novel_id"])
        assert queried_novel["name"] == updated_novel_name, queried_novel
        assert queried_novel["summary"] == f"Novel Updated {suffix}", queried_novel

        updated_outline_title = f"Strict Outline Updated {suffix}"
        updated_outline_content = f"Outline Updated Content {suffix}"
        assert_json_response(
            requests.post(
                f"{API_PREFIX}/archive/update",
                json={
                    "id": chain["outline_id"],
                    "type": "outline",
                    "name": updated_outline_title,
                    "content": updated_outline_content,
                    "novel_id": chain["novel_id"],
                },
                timeout=TIMEOUT,
            )
        )
        queried_outline = get_outline(chain["outline_id"])
        assert queried_outline["title"] == updated_outline_title, queried_outline
        assert queried_outline["summary"] == updated_outline_content, queried_outline

        updated_chapter_title = f"Strict Chapter Updated {suffix}"
        updated_chapter_content = f"Chapter Updated Content {suffix}"
        assert_json_response(
            requests.post(
                f"{API_PREFIX}/archive/update",
                json={
                    "id": chain["chapter_id"],
                    "type": "prose",
                    "name": updated_chapter_title,
                    "content": updated_chapter_content,
                    "outline_id": chain["outline_id"],
                },
                timeout=TIMEOUT,
            )
        )
        queried_chapter = get_chapter(chain["chapter_id"], chain["outline_id"], chain["worldview_id"])
        assert queried_chapter["name"] == updated_chapter_title, queried_chapter
        assert queried_chapter["content"] == updated_chapter_content, queried_chapter

        assert_json_response(
            requests.delete(f"{API_PREFIX}/novels/delete", json={"novel_id": chain["novel_id"]}, timeout=TIMEOUT),
            expected_status=409,
        )
        assert get_novel(chain["novel_id"]) is not None

        assert_json_response(
            requests.delete(f"{API_PREFIX}/worldviews/delete", json={"worldview_id": chain["worldview_id"]}, timeout=TIMEOUT),
        )
        assert get_worldview(chain["worldview_id"]) is None
        assert get_novel(chain["novel_id"]) is not None
        assert get_outline(chain["outline_id"]) is not None
        assert get_chapter(chain["chapter_id"], chain["outline_id"], chain["worldview_id"]) is not None

        assert_json_response(
            requests.delete(f"{API_PREFIX}/worlds/delete", json={"world_id": chain["world_id"]}, timeout=TIMEOUT),
            expected_status=409,
        )
        assert get_world(chain["world_id"]) is not None

        assert_json_response(
            requests.delete(f"{API_PREFIX}/novels/delete", json={"novel_id": chain["novel_id"], "cascade": True}, timeout=TIMEOUT)
        )
        assert get_novel(chain["novel_id"]) is None
        assert get_outline(chain["outline_id"]) is None
        assert get_chapter(chain["chapter_id"], chain["outline_id"], chain["worldview_id"]) is None
        assert_json_response(
            requests.delete(
                f"{API_PREFIX}/worlds/delete",
                json={"world_id": chain["world_id"], "cascade": True},
                timeout=TIMEOUT,
            )
        )
        assert get_world(chain["world_id"]) is None

        chain_two = create_full_chain(f"{suffix}_wv")
        assert_json_response(
            requests.delete(
                f"{API_PREFIX}/worldviews/delete",
                json={"worldview_id": chain_two["worldview_id"], "cascade": True},
                timeout=TIMEOUT,
            )
        )
        assert get_worldview(chain_two["worldview_id"]) is None
        assert get_novel(chain_two["novel_id"]) is not None
        assert get_outline(chain_two["outline_id"]) is not None
        assert get_chapter(chain_two["chapter_id"], chain_two["outline_id"], chain_two["worldview_id"]) is not None
        assert_json_response(
            requests.delete(
                f"{API_PREFIX}/worlds/delete",
                json={"world_id": chain_two["world_id"], "cascade": True},
                timeout=TIMEOUT,
            )
        )
        assert get_world(chain_two["world_id"]) is None

        chain_three = create_full_chain(f"{suffix}_world")
        assert_json_response(
            requests.delete(
                f"{API_PREFIX}/worlds/delete",
                json={"world_id": chain_three["world_id"], "cascade": True},
                timeout=TIMEOUT,
            )
        )
        assert get_world(chain_three["world_id"]) is None
        assert get_worldview(chain_three["worldview_id"]) is None

        chain = {}
    finally:
        if chain.get("world_id"):
            requests.delete(
                f"{API_PREFIX}/worlds/delete",
                json={"world_id": chain["world_id"], "cascade": True},
                timeout=TIMEOUT,
            )


def test_world_hierarchy_invalid_inputs():
    missing_name = assert_json_response(
        requests.post(f"{API_PREFIX}/worlds/create", json={"summary": "missing name"}, timeout=TIMEOUT),
        expected_status=400,
    )
    assert "name" in missing_name.get("error", "").lower(), missing_name

    missing_parent = assert_json_response(
        requests.post(
            f"{API_PREFIX}/novels/create",
            json={"name": "Invalid Novel", "world_id": "world_does_not_exist"},
            timeout=TIMEOUT,
        ),
        expected_status=404,
    )
    assert "not found" in missing_parent.get("error", "").lower(), missing_parent

    bad_archive_type = assert_json_response(
        requests.post(
            f"{API_PREFIX}/archive/update",
            json={"id": "bad_type", "type": "invalid_type", "name": "Invalid"},
            timeout=TIMEOUT,
        ),
        expected_status=400,
    )
    assert "invalid type" in bad_archive_type.get("error", "").lower(), bad_archive_type

    forbidden_full_queries = [
        f"{API_PREFIX}/worldviews/list",
        f"{API_PREFIX}/novels/list",
        f"{API_PREFIX}/outlines/list",
        f"{API_PREFIX}/lore/list",
        f"{API_PREFIX}/world-hierarchy/tree",
    ]
    for url in forbidden_full_queries:
        response = assert_json_response(requests.get(url, timeout=TIMEOUT), expected_status=400)
        error_text = response.get("error", "").lower()
        assert "required" in error_text or "pagination" in error_text, response

    no_pagination = assert_json_response(
        requests.get(f"{API_PREFIX}/worldviews/list", params={"world_id": "world_default"}, timeout=TIMEOUT),
        expected_status=400,
    )
    assert "pagination" in no_pagination.get("error", "").lower(), no_pagination


if __name__ == "__main__":
    test_world_hierarchy_lifecycle()
    test_world_hierarchy_invalid_inputs()
    print("world hierarchy requests tests passed")
