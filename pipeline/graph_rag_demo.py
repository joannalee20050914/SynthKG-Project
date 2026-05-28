import argparse
import csv
import json
import time
from datetime import datetime
from pathlib import Path

import requests

from config import DEFAULT_DATA_FILE, TOP_M, TOP_K, N_HOP, GEMINI_API_KEY, GEMINI_MODEL
from graph_loader import load_graph, build_indices
from retriever import (
    retrieve_top_m,
    extract_query_entities,
    build_subgraph,
    n_hop_traversal,
    select_top_k_chunks,
)
from reranker import rerank


# ===== Demo log files: 寫到專案根目錄，避免和正式版混在一起 =====
DEMO_EXPERIMENT_CSV = Path(__file__).resolve().parent.parent / "graph_rag_demo_experiment_logs.csv"
DEMO_ERROR_CSV = Path(__file__).resolve().parent.parent / "graph_rag_demo_error_logs.csv"
DEMO_TEXT_LOG = Path(__file__).resolve().parent.parent / "graph_rag_demo_run_log.txt"


def resolve_graph_path(graph_arg: str | None) -> Path:
    base_dir = Path(__file__).resolve().parent
    graph_file = graph_arg if graph_arg else DEFAULT_DATA_FILE
    graph_path = Path(graph_file)

    if graph_path.is_absolute():
        return graph_path

    return (base_dir / graph_path).resolve()


def append_demo_experiment_log_csv(log_path: str, row: dict):
    file_exists = Path(log_path).exists()

    with open(log_path, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "timestamp",
                "query",
                "graph_file",
                "status",
                "rerank_input_tokens",
                "rerank_output_tokens",
                "rerank_total_tokens",
                "generation_input_tokens",
                "generation_output_tokens",
                "generation_total_tokens",
                "execution_time_seconds",
                "fallback_used",
                "generation_finish_reason",
                "candidate_count",
                "final_count",
            ],
        )
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def append_demo_error_log_csv(log_path: str, row: dict):
    file_exists = Path(log_path).exists()

    with open(log_path, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "timestamp",
                "query",
                "graph_file",
                "error_type",
                "error_message",
            ],
        )
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def append_demo_text_log(log_path: str, message: str):
    with open(log_path, "a", encoding="utf-8-sig") as f:
        f.write(message + "\n")


def build_generation_context(final_results: list[dict]) -> str:
    lines = []
    for idx, item in enumerate(final_results, start=1):
        lines.append(f"[Evidence {idx}]")
        lines.append(f"Proposition: {item['proposition']}")
        lines.append("Triplets:")
        for t in item["triplets"]:
            lines.append(f"- ({t['head']}, {t['relation']}, {t['tail']})")
        lines.append("")
    return "\n".join(lines)


def generate_answer(query: str, final_results: list[dict]) -> tuple[str, dict, str]:
    """
    回傳:
    - generated_answer
    - generation_usage_info
    - generation_finish_reason
    """
    if not GEMINI_API_KEY:
        return (
            "[Generation skipped] GEMINI_API_KEY is empty.",
            {
                "input_tokens": "",
                "output_tokens": "",
                "total_tokens": ""
            },
            "",
        )

    context = build_generation_context(final_results)

    prompt = f"""
You are a question answering assistant.
Answer the user's question only based on the evidence below.

Question:
{query}

Evidence:
{context}

Instructions:
1. Provide a complete answer in 1-2 full sentences.
2. Mention the specific exercises or actions if they appear in the evidence.
3. Do not mention proposition IDs.
4. Do not say "based on the evidence".
5. If the evidence is insufficient, say so briefly.
"""

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": 150,
            "thinkingConfig": {"thinkingBudget": 0},
        },
    }

    response = requests.post(url, json=payload, timeout=60)
    response.raise_for_status()
    data = response.json()

    usage = data.get("usageMetadata", {})
    generation_usage_info = {
        "input_tokens": usage.get("promptTokenCount", ""),
        "output_tokens": usage.get("candidatesTokenCount", ""),
        "total_tokens": usage.get("totalTokenCount", "")
    }

    candidates = data.get("candidates", [])
    if not candidates:
        return (
            "[Generation failed] No candidates returned.",
            generation_usage_info,
            "",
        )

    generation_finish_reason = candidates[0].get("finishReason", "")

    content = candidates[0].get("content", {})
    parts = content.get("parts", [])
    if not parts or "text" not in parts[0]:
        return (
            "[Generation failed] Empty content returned.",
            generation_usage_info,
            generation_finish_reason,
        )

    generated_answer = parts[0]["text"].strip()
    return generated_answer, generation_usage_info, generation_finish_reason


def main():
    start_time = time.perf_counter()
    query = ""
    graph_path = ""

    try:
        parser = argparse.ArgumentParser(description="Graph RAG Demo")
        parser.add_argument(
            "--graph",
            type=str,
            default=None,
            help="Path to graph JSON file",
        )
        parser.add_argument(
            "--query",
            type=str,
            default=None,
            help="Optional query string. If not provided, input() will be used.",
        )
        args = parser.parse_args()

        graph_path = resolve_graph_path(args.graph)
        data = load_graph(str(graph_path))
        propositions, entity_to_props, prop_to_entities = build_indices(data)

        if args.query:
            query = args.query.strip()
        else:
            query = input("Please enter your query: ").strip()

        top_props = retrieve_top_m(query, propositions, TOP_M)
        all_entities = list(entity_to_props.keys())
        query_entities = extract_query_entities(query, all_entities)
        subgraph = build_subgraph(top_props, prop_to_entities, entity_to_props)

        if query_entities:
            candidate_prop_ids = n_hop_traversal(query_entities, subgraph, prop_to_entities, N_HOP)
            candidates = select_top_k_chunks(candidate_prop_ids, subgraph, query, TOP_K)

            if len(candidates) < TOP_K:
                existing_ids = {item["prop_id"] for item in candidates}
                for prop in top_props:
                    if prop["prop_id"] not in existing_ids:
                        candidates.append(prop)
                    if len(candidates) >= TOP_K:
                        break
        else:
            candidates = top_props[:TOP_K]

        final_results, rerank_usage_info, _, fallback_used = rerank(query, candidates, TOP_K)
        generated_answer, generation_usage_info, generation_finish_reason = generate_answer(query, final_results)

        elapsed_sec = time.perf_counter() - start_time

        print("\n=== Answer ===")
        print(generated_answer)

        print("\n=== Supporting Evidence ===")
        for idx, item in enumerate(final_results, start=1):
            print(f"[{idx}] {item['proposition']}")

        append_demo_experiment_log_csv(
            str(DEMO_EXPERIMENT_CSV),
            {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "query": query,
                "graph_file": str(graph_path),
                "status": "success",
                "rerank_input_tokens": rerank_usage_info.get("input_tokens", ""),
                "rerank_output_tokens": rerank_usage_info.get("output_tokens", ""),
                "rerank_total_tokens": rerank_usage_info.get("total_tokens", ""),
                "generation_input_tokens": generation_usage_info.get("input_tokens", ""),
                "generation_output_tokens": generation_usage_info.get("output_tokens", ""),
                "generation_total_tokens": generation_usage_info.get("total_tokens", ""),
                "execution_time_seconds": f"{elapsed_sec:.3f}",
                "fallback_used": fallback_used,
                "generation_finish_reason": generation_finish_reason,
                "candidate_count": len(candidates),
                "final_count": len(final_results),
            },
        )

        append_demo_text_log(
            str(DEMO_TEXT_LOG),
            "=" * 80
        )
        append_demo_text_log(
            str(DEMO_TEXT_LOG),
            f"Time: {datetime.now().isoformat(timespec='seconds')}"
        )
        append_demo_text_log(
            str(DEMO_TEXT_LOG),
            f"Query: {query}"
        )
        append_demo_text_log(
            str(DEMO_TEXT_LOG),
            f"Graph file: {graph_path}"
        )
        append_demo_text_log(
            str(DEMO_TEXT_LOG),
            f"Answer: {generated_answer}"
        )
        append_demo_text_log(
            str(DEMO_TEXT_LOG),
            f"Execution time: {elapsed_sec:.3f} sec"
        )
        append_demo_text_log(
            str(DEMO_TEXT_LOG),
            f"Rerank total tokens: {rerank_usage_info.get('total_tokens', '')}"
        )
        append_demo_text_log(
            str(DEMO_TEXT_LOG),
            f"Generation total tokens: {generation_usage_info.get('total_tokens', '')}"
        )
        append_demo_text_log(
            str(DEMO_TEXT_LOG),
            f"Generation finish reason: {generation_finish_reason}"
        )
        append_demo_text_log(
            str(DEMO_TEXT_LOG),
            "Supporting Evidence:"
        )
        for idx, item in enumerate(final_results[:3], start=1):
            append_demo_text_log(
                str(DEMO_TEXT_LOG),
                f"[{idx}] {item['proposition']}"
            )
        append_demo_text_log(str(DEMO_TEXT_LOG), "")

    except Exception as e:
        append_demo_error_log_csv(
            str(DEMO_ERROR_CSV),
            {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "query": query,
                "graph_file": str(graph_path),
                "error_type": type(e).__name__,
                "error_message": str(e),
            },
        )

        append_demo_text_log(
            str(DEMO_TEXT_LOG),
            "=" * 80
        )
        append_demo_text_log(
            str(DEMO_TEXT_LOG),
            f"Time: {datetime.now().isoformat(timespec='seconds')}"
        )
        append_demo_text_log(
            str(DEMO_TEXT_LOG),
            f"[ERROR] {type(e).__name__}: {e}"
        )

        print(f"[Error] {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()