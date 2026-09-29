"""Check the offline 32-post fixture through the public HTTP contract."""
import json
import sys
import urllib.request

base = sys.argv[1].rstrip("/")


def get(path):
    with urllib.request.urlopen(base + path, timeout=30) as response:
        return json.load(response)


def post(path, body):
    request = urllib.request.Request(base + path, json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


assert get("/healthz")["status"] == "ok"
assert get("/readyz")["status"] == "ready"
tree = get("/v1/tree")
assert tree["embedding"]["dimension"] == 256
assert sum(leaf["post_count"] for leaf in tree["leaves"]) == 32
query = {"embedding_space_id": tree["embedding_space_id"], "vector": [1.] + [0.] * 255}
near = post("/v1/nearby", query)
assert near["returned"] == 25
assert len({item["identity"]["native_id"] for item in near["items"]}) == 25
assert all(len(item["delta"]) == 256 for item in near["items"])
terms = post("/v1/terms", query)
assert terms["covered"] == 32
assert any(term["term"] == "julia" and term["count"] == 32 for term in terms["terms"])
print(json.dumps({"store_id": tree["store_id"], "posts": 32, "nearby": 25, "keywords": True}))
