import json
import re
import requests
from config import RERANK_PROVIDER, GEMINI_API_KEY, GEMINI_MODEL
from retriever import score_texts_by_embedding


def simple_rerank(query: str, candidates: list[dict], top_k: int) -> list[dict]:
    if not candidates:
        return []

    candidate_texts = [c["proposition"] for c in candidates]
    scores = score_texts_by_embedding(query, candidate_texts)

    scored = list(zip(scores, candidates))
    scored.sort(key=lambda x: x[0], reverse=True)

    return [item[1] for item in scored[:top_k]]


def build_rerank_prompt(query: str, candidates: list[dict], top_k: int) -> str:
    lines = []
    lines.append("You are a retrieval reranker.")
    lines.append(f"Question: {query}")
    lines.append("")
    lines.append("Candidate propositions:")
    for c in candidates:
        lines.append(f"{c['prop_id']}: {c['proposition']}")
    lines.append("")
    lines.append(f"Select the top {top_k} most relevant propositions for answering the question.")
    lines.append("Return ONLY valid JSON in this exact format:")
    lines.append('{ "selected_prop_ids": ["P1", "P2"] }')
    lines.append("Do not add explanation. Do not use markdown code fences.")
    return "\n".join(lines)


def extract_json_from_text(text: str) -> dict | None:
    """
    從 Gemini 回傳文字中，盡量抓出 JSON
    """
    if not text:
        return None

    # 去掉 markdown code fence
    cleaned = text.strip()
    cleaned = cleaned.replace("```json", "").replace("```", "").strip()

    # 先嘗試整段直接 loads
    try:
        return json.loads(cleaned)
    except Exception:
        pass

    # 再找第一個 {...}
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            pass

    return None


def extract_prop_ids_from_text(text: str) -> list[str]:
    """
    如果 Gemini 沒回合法 JSON，就直接從文字抓 Pxx
    """
    return re.findall(r"P\d+", text)


def gemini_rerank(query: str, candidates: list[dict], top_k: int) -> list[dict]:
    if not GEMINI_API_KEY:
        print("[Warning] GEMINI_API_KEY is empty. Fallback to simple_rerank.")
        return simple_rerank(query, candidates, top_k)

    prompt = build_rerank_prompt(query, candidates, top_k)

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"

    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt}
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0
        }
    }

    try:
        response = requests.post(url, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()

        text = data["candidates"][0]["content"]["parts"][0]["text"]

        print("\n[Debug] Raw Gemini response:")
        print(text)

        parsed = extract_json_from_text(text)

        if parsed and "selected_prop_ids" in parsed:
            selected_ids = parsed.get("selected_prop_ids", [])
        else:
            selected_ids = extract_prop_ids_from_text(text)

        selected_map = {c["prop_id"]: c for c in candidates}
        reranked = [selected_map[pid] for pid in selected_ids if pid in selected_map]

        # 如果 Gemini 回太少，就補簡單 rerank
        if len(reranked) < top_k:
            fallback = simple_rerank(query, candidates, top_k)
            used = {item["prop_id"] for item in reranked}
            for item in fallback:
                if item["prop_id"] not in used:
                    reranked.append(item)
                if len(reranked) >= top_k:
                    break

        return reranked[:top_k]

    except Exception as e:
        print(f"[Warning] Gemini reranking failed: {e}")
        return simple_rerank(query, candidates, top_k)


def rerank(query: str, candidates: list[dict], top_k: int) -> list[dict]:
    if RERANK_PROVIDER == "gemini":
        return gemini_rerank(query, candidates, top_k)
    return simple_rerank(query, candidates, top_k)