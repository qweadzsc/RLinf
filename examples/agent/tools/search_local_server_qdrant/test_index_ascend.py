import json
from qdrant_client import AsyncQdrantClient, QdrantClient

from tqdm import tqdm

CORPUS = "/home/infiniai/zhouyongkang/datasets/ASearcher-Local-Knowledge/wiki_corpus.jsonl"
QDRANT_URL = "http://localhost:6333"
COLLECTION = "wiki_collection"

# 1. 统计预期写入数量：build_index.py 会跳过空 contents
# client = QdrantClient(url=QDRANT_URL)
client = QdrantClient(url=QDRANT_URL, prefer_grpc=True, timeout=60)
info = client.get_collection(COLLECTION)

expected = 26134257
# with open(CORPUS, encoding="utf-8") as f:
#     for line in tqdm(f, desc="Counting documents"):
#         doc = json.loads(line)
#         if doc.get("contents", "").strip():
#             expected += 1

print("status:", info.status)
print("points:", info.points_count)
print("expected:", expected)
print("dimension:", info.config.params.vectors.size)

assert str(info.status).lower().endswith("green")
assert info.points_count == expected, (
    f"索引不完整：实际 {info.points_count}，预期 {expected}"
)

# 2. 抽样确认 payload、ID、向量维度存在
points, _ = client.scroll(
    collection_name=COLLECTION,
    limit=5,
    with_payload=True,
    with_vectors=True,
)

for point in tqdm(points, desc="Checking index structure"):
    assert point.payload and point.payload.get("contents", "").strip()
    assert len(point.vector) == info.config.params.vectors.size
    print("id:", point.id, "text:", point.payload["contents"][:100])

print("Index structural checks passed.")
