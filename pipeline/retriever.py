import re
import numpy as np
from collections import deque
from sentence_transformers import SentenceTransformer
from config import EMBED_MODEL


_model = None


def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBED_MODEL)
    return _model


def tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-zA-Z0-9]+", text.lower()))


def embed_texts(texts: list[str]) -> np.ndarray:
    model = get_model()
    return model.encode(texts, normalize_embeddings=True)


def score_texts_by_embedding(query: str, texts: list[str]) -> list[float]:
    """
    用 embedding cosine similarity 計算 query 與 texts 的相似度
    因為 normalize_embeddings=True，所以 dot product 就等於 cosine similarity
    """
    all_embeddings = embed_texts([query] + texts)
    query_emb = all_embeddings[0]
    text_embs = all_embeddings[1:]
    scores = text_embs @ query_emb
    return scores.tolist()


def retrieve_top_m(query: str, propositions: list[dict], top_m: int) -> list[dict]:
    prop_texts = [p["proposition"] for p in propositions]
    scores = score_texts_by_embedding(query, prop_texts)

    scored = list(zip(scores, propositions))
    scored.sort(key=lambda x: x[0], reverse=True)

    return [item[1] for item in scored[:top_m]]


def normalize_token(token: str) -> str:
    """
    很簡單的正規化：
    - 小寫
    - 去掉尾端單複數 s
    """
    token = token.lower().strip()
    if token.endswith("s") and len(token) > 3:
        token = token[:-1]
    return token


def normalize_tokens(tokens: set[str]) -> set[str]:
    return {normalize_token(t) for t in tokens}


def extract_query_entities(query: str, all_entities: list[str]) -> list[str]:
    """
    收緊版 query entity matching：
    1. 完整短語 match 優先
    2. 多詞 entity 至少 2 個 token overlap
    3. 單詞 entity 只接受正規化後完全相同
    """
    query_lower = query.lower()
    q_tokens_raw = tokenize(query)
    q_tokens = normalize_tokens(q_tokens_raw)

    matched = []

    for entity in all_entities:
        entity_lower = entity.lower()
        e_tokens_raw = tokenize(entity)
        e_tokens = normalize_tokens(e_tokens_raw)

        # 規則 1：完整短語包含
        if entity_lower in query_lower:
            matched.append(entity)
            continue

        # 規則 2：多詞 entity 至少 2 個 overlap
        if len(e_tokens) >= 2:
            overlap = q_tokens & e_tokens
            if len(overlap) >= 2:
                matched.append(entity)
                continue

        # 規則 3：單詞 entity 只接受完全相同
        if len(e_tokens) == 1:
            single = next(iter(e_tokens))
            if single in q_tokens:
                matched.append(entity)

    return list(dict.fromkeys(matched))


def build_subgraph(top_props: list[dict], prop_to_entities: dict[str, set], entity_to_props: dict[str, set]) -> dict:
    subgraph = {"props": {}, "entities": {}}

    for prop in top_props:
        prop_id = prop["prop_id"]
        subgraph["props"][prop_id] = prop

        for entity in prop_to_entities[prop_id]:
            if entity not in subgraph["entities"]:
                subgraph["entities"][entity] = []
            subgraph["entities"][entity].append(prop_id)

    return subgraph


def n_hop_traversal(query_entities: list[str], subgraph: dict, prop_to_entities: dict[str, set], n_hop: int) -> list[str]:
    visited_entities = set()
    visited_props = set()
    queue = deque()

    for entity in query_entities:
        if entity in subgraph["entities"]:
            queue.append(("entity", entity, 0))
            visited_entities.add(entity)

    while queue:
        node_type, node_value, depth = queue.popleft()

        if depth >= n_hop:
            continue

        if node_type == "entity":
            connected_props = subgraph["entities"].get(node_value, [])
            for prop_id in connected_props:
                if prop_id not in visited_props:
                    visited_props.add(prop_id)
                    queue.append(("prop", prop_id, depth + 1))

        elif node_type == "prop":
            connected_entities = prop_to_entities.get(node_value, set())
            for entity in connected_entities:
                if entity not in visited_entities and entity in subgraph["entities"]:
                    visited_entities.add(entity)
                    queue.append(("entity", entity, depth + 1))

    return list(visited_props)


def select_top_k_chunks(prop_ids: list[str], subgraph: dict, query: str, top_k: int) -> list[dict]:
    candidates = [subgraph["props"][pid] for pid in prop_ids if pid in subgraph["props"]]
    if not candidates:
        return []

    candidate_texts = [c["proposition"] for c in candidates]
    scores = score_texts_by_embedding(query, candidate_texts)

    scored = list(zip(scores, candidates))
    scored.sort(key=lambda x: x[0], reverse=True)

    return [item[1] for item in scored[:top_k]]