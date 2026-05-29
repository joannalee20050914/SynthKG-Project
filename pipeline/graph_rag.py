import argparse
import csv
import time
import json
from datetime import datetime
from pathlib import Path

import requests

from config import (
    DEFAULT_DATA_FILE,
    TOP_M,
    TOP_K,
    N_HOP,
    RERANK_PROVIDER,
    GEMINI_API_KEY,
    GEMINI_MODEL,
)
from graph_loader import load_graph, build_indices
from retriever import (
    retrieve_top_m,
    extract_query_entities,
    build_subgraph,
    n_hop_traversal,
    select_top_k_chunks,
)
from reranker import rerank
ERROR_LOG_FILE = Path(__file__).resolve().parent.parent / "error_logs.csv"
METRICS_FILE = Path(__file__).resolve().parent.parent / "query_metrics.csv"
EXPERIMENT_LOG_FILE = Path(__file__).resolve().parent.parent / "experiment_logs.txt"


def append_experiment_log(
    query,
    graph_file,
    final_results,
    rerank_usage_info,
    generation_usage_info,
    generated_answer,
    generation_finish_reason,
    raw_generation_response,
    elapsed_sec,
    fallback_used
):
    with open(EXPERIMENT_LOG_FILE, "a", encoding="utf-8-sig") as f:
        f.write("=" * 80 + "\n")
        f.write(f"Time: {datetime.now().isoformat(timespec='seconds')}\n")
        f.write(f"Query: {query}\n")
        f.write(f"Graph file: {graph_file}\n")
        f.write(f"Elapsed time: {elapsed_sec:.3f} sec\n")

        f.write("[Rerank Usage]\n")
        f.write(f"Input tokens: {rerank_usage_info.get('input_tokens', '')}\n")
        f.write(f"Output tokens: {rerank_usage_info.get('output_tokens', '')}\n")
        f.write(f"Total tokens: {rerank_usage_info.get('total_tokens', '')}\n")
        f.write(f"Fallback used: {fallback_used}\n")

        f.write("[Generation Usage]\n")
        f.write(f"Input tokens: {generation_usage_info.get('input_tokens', '')}\n")
        f.write(f"Output tokens: {generation_usage_info.get('output_tokens', '')}\n")
        f.write(f"Total tokens: {generation_usage_info.get('total_tokens', '')}\n")
        f.write(f"Finish reason: {generation_finish_reason}\n")

        f.write("\nGenerated Answer:\n")
        f.write(generated_answer + "\n\n")
        f.write("\nRaw Generation Response:\n")
        f.write(raw_generation_response + "\n")

        f.write("Final Results:\n")
        for idx, item in enumerate(final_results, start=1):
            f.write(f"  [{idx}] {item['prop_id']} | {item['proposition']}\n")

        f.write("\n")


def resolve_graph_path(graph_arg: str | None) -> Path:
    """
    決定要讀哪個 graph 檔案
    - 如果有給 --graph，就用它
    - 如果沒給，就用 config.py 裡的 DEFAULT_DATA_FILE
    """
    base_dir = Path(__file__).resolve().parent  # pipeline 資料夾
    graph_file = graph_arg if graph_arg else DEFAULT_DATA_FILE
    graph_path = Path(graph_file)

    if graph_path.is_absolute():
        return graph_path

    # 相對路徑一律以 main.py 所在的 pipeline/ 為基準
    return (base_dir / graph_path).resolve()


def append_error_log(query, graph_file, error_type, error_message, raw_response="", provider=""):
    file_exists = ERROR_LOG_FILE.exists()

    with open(ERROR_LOG_FILE, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow([
                "timestamp",
                "query",
                "graph_file",
                "provider",
                "error_type",
                "error_message",
                "raw_response"
            ])

        writer.writerow([
            datetime.now().isoformat(timespec="seconds"),
            query,
            str(graph_file),
            provider,
            error_type,
            error_message,
            raw_response
        ])


def append_metrics(query, graph_file, elapsed_sec, input_tokens="", output_tokens="", fallback_used="", candidate_count="", final_count=""):
    file_exists = METRICS_FILE.exists()

    with open(METRICS_FILE, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow([
                "timestamp",
                "query",
                "graph_file",
                "elapsed_sec",
                "input_tokens",
                "output_tokens",
                "fallback_used",
                "candidate_count",
                "final_count"
            ])

        writer.writerow([
            datetime.now().isoformat(timespec="seconds"),
            query,
            str(graph_file),
            f"{elapsed_sec:.3f}",
            input_tokens,
            output_tokens,
            fallback_used,
            candidate_count,
            final_count
        ])


def build_generation_context(final_results: list[dict]) -> str:
    """
    把 reranking 後留下來的 propositions / triplets 組成可給 LLM 的 context
    """
    lines = []
    for idx, item in enumerate(final_results, start=1):
        lines.append(f"[Evidence {idx}]")
        lines.append(f"Proposition: {item['proposition']}")
        lines.append("Triplets:")
        for t in item["triplets"]:
            lines.append(f"- ({t['head']}, {t['relation']}, {t['tail']})")
        lines.append("")
    return "\n".join(lines)


def generate_answer(query: str, final_results: list[dict]) -> tuple[str, dict, str, str]:
    """
    用 reranked evidence 當 context，讓 Gemini 生成自然語言答案
    回傳：
      - answer_text
      - usage_info
      - raw_generation_response
      - finish_reason
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
            ""
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
Instructions:
1. Provide a complete answer in 1-2 full sentences.
2. Mention the specific exercises or actions if they appear in the evidence.
3. Do not mention proposition IDs.
4. Do not say "based on the evidence".
5. If the evidence is insufficient, say so briefly.
"""

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
            "temperature": 0.2,
            "maxOutputTokens": 150,
            "thinkingConfig": {
                "thinkingBudget": 0
            }
        }
    }

    try:
        response = requests.post(url, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()

        # 先把整包原始回應轉成字串，方便 debug / logging
        raw_generation_response = json.dumps(data, ensure_ascii=False, indent=2)

        usage = data.get("usageMetadata", {})
        usage_info = {
            "input_tokens": usage.get("promptTokenCount", ""),
            "output_tokens": usage.get("candidatesTokenCount", ""),
            "total_tokens": usage.get("totalTokenCount", "")
        }

        candidates = data.get("candidates", [])
        if not candidates:
            return (
                "[Generation failed] No candidates returned.",
                usage_info,
                raw_generation_response,
                ""
            )

        finish_reason = candidates[0].get("finishReason", "")

        content = candidates[0].get("content", {})
        parts = content.get("parts", [])

        if not parts or "text" not in parts[0]:
            return (
                f"[Generation failed] Empty content. finishReason={finish_reason}",
                usage_info,
                raw_generation_response,
                finish_reason
            )

        answer_text = parts[0]["text"].strip()

        return answer_text, usage_info, raw_generation_response, finish_reason

    except Exception as e:
        return (
            f"[Generation failed] {type(e).__name__}: {e}",
            {
                "input_tokens": "",
                "output_tokens": "",
                "total_tokens": ""
            },
            "",
            ""
        )


def main():
    start_time = time.perf_counter()
    query = ""
    graph_path = ""

    try: 
        parser = argparse.ArgumentParser(description="Proposition-Entity Graph Retriever Demo")
        parser.add_argument(
            "--graph",
            type=str,
            default=None,
            help="Path to graph JSON file, e.g. ../extraction/outputs/graph_zeroshot.json",
        )
        parser.add_argument(
            "--query",
            type=str,
            default=None,
            help="Optional query string. If not provided, input() will be used.",
        )

        args = parser.parse_args()

        graph_path = resolve_graph_path(args.graph)

        print("=== Proposition-Entity Graph Retriever Demo ===")
        print(f"[Debug] Using graph file: {graph_path}")

        data = load_graph(str(graph_path))
        propositions, entity_to_props, prop_to_entities = build_indices(data)

        print(f"\n[Debug] Total propositions loaded: {len(propositions)}")
        print(f"[Debug] Total unique entities loaded: {len(entity_to_props)}")

        if args.query:
            query = args.query.strip()
            print(f"\n[Debug] Query from argument: {query}")
        else:
            query = input("\nPlease enter your query: ").strip()

        # Step 1: retrieve top-M propositions
        top_props = retrieve_top_m(query, propositions, TOP_M)

        print("\n=== [Debug] Top-M Retrieved Propositions ===")
        for idx, prop in enumerate(top_props, start=1):
            print(f"[{idx}] {prop['prop_id']} | {prop['proposition']}")

        # Query entity extraction
        all_entities = list(entity_to_props.keys())
        query_entities = extract_query_entities(query, all_entities)

        print("\n=== [Debug] Query Entities ===")
        if query_entities:
            print(query_entities)
        else:
            print("No query entities matched.")

        # Step 2: build subgraph
        subgraph = build_subgraph(top_props, prop_to_entities, entity_to_props)

        print("\n=== [Debug] Subgraph Summary ===")
        print(f"Subgraph propositions: {len(subgraph['props'])}")
        print(f"Subgraph entities: {len(subgraph['entities'])}")

        # Step 3: N-hop traversal
        if query_entities:
            candidate_prop_ids = n_hop_traversal(query_entities, subgraph, prop_to_entities, N_HOP)

            print("\n=== [Debug] Candidate Proposition IDs after N-hop Traversal ===")
            print(candidate_prop_ids)

            candidates = select_top_k_chunks(candidate_prop_ids, subgraph, query, TOP_K)

            # 如果 traversal 後候選太少，也可以視需要加 fallback
            if len(candidates) < TOP_K:
                existing_ids = {item["prop_id"] for item in candidates}
                for prop in top_props:
                    if prop["prop_id"] not in existing_ids:
                        candidates.append(prop)
                    if len(candidates) >= TOP_K:
                        break
        else:
            candidates = top_props[:TOP_K]

        print("\n=== [Debug] Candidates Before Reranking ===")
        for idx, item in enumerate(candidates, start=1):
            print(f"[{idx}] {item['prop_id']} | {item['proposition']}")

        # Step 4: rerank
        final_results, rerank_usage_info, raw_response_text, fallback_used = rerank(query, candidates, TOP_K)

        print("\n=== Final Results ===")
        for idx, item in enumerate(final_results, start=1):
            print(f"\n[{idx}] Prop ID: {item['prop_id']}")
            print(f"Post ID: {item['source_post_id']}")
            print("Triplets:")
            for t in item["triplets"]:
                print(f"  - ({t['head']}, {t['relation']}, {t['tail']})")
            print(f"Proposition: {item['proposition']}")

        # Step 5: generation
        generated_answer, generation_usage_info, raw_generation_response, generation_finish_reason = generate_answer(query, final_results)

        print("\n=== Generated Answer ===")
        print(generated_answer)
        print(f"[Debug] Generation finish reason: {generation_finish_reason}")

        print("\n[Debug] Raw Generation response:")
        print(raw_generation_response)

        elapsed_sec = time.perf_counter() - start_time
        print(f"\n[Debug] Elapsed time: {elapsed_sec:.3f} sec")

        # rerank token
        print(f"[Debug] Rerank input tokens: {rerank_usage_info.get('input_tokens', '')}")
        print(f"[Debug] Rerank output tokens: {rerank_usage_info.get('output_tokens', '')}")
        print(f"[Debug] Rerank total tokens: {rerank_usage_info.get('total_tokens', '')}")
        print(f"[Debug] Fallback used: {fallback_used}")

        # generation token
        print(f"[Debug] Generation input tokens: {generation_usage_info.get('input_tokens', '')}")
        print(f"[Debug] Generation output tokens: {generation_usage_info.get('output_tokens', '')}")
        print(f"[Debug] Generation total tokens: {generation_usage_info.get('total_tokens', '')}")

        append_metrics(
            query=query,
            graph_file=graph_path,
            elapsed_sec=elapsed_sec,
            input_tokens=rerank_usage_info.get("input_tokens", ""),
            output_tokens=rerank_usage_info.get("output_tokens", ""),
            fallback_used=fallback_used,
            candidate_count=len(candidates),
            final_count=len(final_results)
        )

        append_experiment_log(
            query=query,
            graph_file=graph_path,
            final_results=final_results,
            rerank_usage_info=rerank_usage_info,
            generation_usage_info=generation_usage_info,
            generated_answer=generated_answer,
            generation_finish_reason=generation_finish_reason,
            raw_generation_response=raw_generation_response,
            elapsed_sec=elapsed_sec,
            fallback_used=fallback_used
        )

    except Exception as e:
        append_error_log(
            query=query,
            graph_file=graph_path,
            error_type=type(e).__name__,
            error_message=str(e),
            raw_response="",
            provider=RERANK_PROVIDER
        )
        print(f"[Error] {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()