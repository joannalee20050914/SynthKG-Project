import json
import re
import requests
import time
from config import RERANK_PROVIDER, GEMINI_API_KEY, GEMINI_MODEL
from retriever import score_texts_by_embedding


def simple_rerank(query: str, candidates: list[dict], top_k: int):
    if not candidates:
        return [], {
            "input_tokens": "",
            "output_tokens": "",
            "total_tokens": ""
        }, "", True

    candidate_texts = [c["proposition"] for c in candidates]
    scores = score_texts_by_embedding(query, candidate_texts)

    scored = list(zip(scores, candidates))
    scored.sort(key=lambda x: x[0], reverse=True)

    results = [item[1] for item in scored[:top_k]]

    return results, {
        "input_tokens": "",
        "output_tokens": "",
        "total_tokens": ""
    }, "", True


def build_rerank_prompt(query: str, candidates: list[dict], top_k: int) -> str:
    lines = []
    lines.append("You are a retrieval reranker for question answering.")
    lines.append("Your job is to rank the candidate propositions by how directly they answer the user's question.")
    lines.append("")
    lines.append(f"Question: {query}")
    lines.append("")
    lines.append("Candidate propositions:")
    for c in candidates:
        lines.append(f"{c['prop_id']}: {c['proposition']}")
    lines.append("")
    lines.append("Ranking rules:")
    lines.append("1. Prefer propositions that directly answer the question.")
    lines.append("2. Prefer core explanations over secondary consequences.")
    lines.append("3. Prefer propositions that are specific to the question focus.")
    lines.append("4. Only include propositions that are truly relevant.")
    lines.append("")
    lines.append(f"Return ONLY the top {top_k} proposition IDs in valid JSON.")
    lines.append('Return exactly this format: {"selected_prop_ids": ["P1", "P2"]}')
    lines.append("Do not add explanation. Do not use markdown code fences.")
    return "\n".join(lines)


def extract_json_from_text(text: str) -> dict | None:
    if not text:
        return None

    cleaned = text.strip()
    cleaned = cleaned.replace("```json", "").replace("```", "").strip()

    try:
        return json.loads(cleaned)
    except Exception:
        pass

    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            pass

    return None


def extract_prop_ids_from_text(text: str) -> list[str]:
    return re.findall(r"P\d+", text)


def gemini_rerank(query: str, candidates: list[dict], top_k: int):
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
        "temperature": 0,
        "responseMimeType": "application/json",
        "responseSchema": {
            "type": "object",
            "properties": {
                "selected_prop_ids": {
                    "type": "array",
                    "items": {
                        "type": "string"
                    }
                }
            },
            "required": ["selected_prop_ids"]
        },
        "thinkingConfig": {
            "thinkingBudget": 0
        },
        "maxOutputTokens": 64
    }
}

    try:
        # retry version
        last_exception = None

        for attempt in range(3):
            try:
                response = requests.post(url, json=payload, timeout=60)
                response.raise_for_status()
                data = response.json()
                break
            except requests.exceptions.HTTPError as e:
                last_exception = e
                status_code = e.response.status_code if e.response is not None else None

                # 只對 503 做重試
                if status_code == 503 and attempt < 2:
                    wait_sec = 2 * (attempt + 1)
                    print(f"[Warning] Gemini API returned 503. Retrying in {wait_sec} seconds...")
                    time.sleep(wait_sec)
                    continue
                raise
            except Exception as e:
                last_exception = e
                raise
        else:
            raise last_exception
        # retry end

        usage = data.get("usageMetadata", {})
        usage_info = {
            "input_tokens": usage.get("promptTokenCount", ""),
            "output_tokens": usage.get("candidatesTokenCount", ""),
            "total_tokens": usage.get("totalTokenCount", "")
        }

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

        fallback_used = False

        if len(reranked) < top_k:
            fallback_used = True
            fallback_results, _, _, _ = simple_rerank(query, candidates, top_k)
            used = {item["prop_id"] for item in reranked}
            for item in fallback_results:
                if item["prop_id"] not in used:
                    reranked.append(item)
                if len(reranked) >= top_k:
                    break

        return reranked[:top_k], usage_info, text, fallback_used

    except Exception as e:
        print(f"[Warning] Gemini reranking failed: {e}")
        results, usage_info, raw_text, fallback_used = simple_rerank(query, candidates, top_k)
        return results, usage_info, raw_text, fallback_used


def rerank(query: str, candidates: list[dict], top_k: int):
    if RERANK_PROVIDER == "gemini":
        return gemini_rerank(query, candidates, top_k)
    return simple_rerank(query, candidates, top_k)