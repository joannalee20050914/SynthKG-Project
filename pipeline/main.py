import json
import csv
import time
import sys
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import jsonlines
from extraction.extractor import extract_triplets   


def read_jsonl(input_path: str) -> list[dict]:
    records = []
    with jsonlines.open(input_path, "r") as reader:
        for obj in reader:
            records.append(obj)
    return records


def save_graph(triplets: list[dict], output_path: str):
    data = {"triplets": triplets}
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def append_experiment_log_csv(log_path: str, row: dict):
    file_exists = Path(log_path).exists()

    with open(log_path, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "post_id",
                "strategy",
                "model",
                "input_tokens",
                "output_tokens",
                "execution_time_seconds",
                "status",
            ],
        )
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def append_error_log_csv(log_path: str, row: dict):
    file_exists = Path(log_path).exists()

    with open(log_path, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "post_id",
                "error_type",
                "raw_response",
            ],
        )
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def append_text_log(log_path: str, message: str):
    with open(log_path, "a", encoding="utf-8-sig") as f:
        f.write(message + "\n")


def validate_triplets(post_id: str, triplets: list[dict]) -> list[dict]:
    validated = []

    for triplet in triplets:
        # 沿用 B 的 source_post_id，但由你這邊做驗證
        if triplet.get("source_post_id") != post_id:
            triplet["source_post_id"] = post_id
        validated.append(triplet)

    return validated


def process_one_record(
    post_id: str,
    text: str,
    strategy: str,
    experiment_csv: str,
    error_csv: str,
    text_log: str,
) -> list[dict]:
    start_time = time.perf_counter()

    try:
        result = extract_triplets(post_id, text, strategy)

        triplets = result.get("triplets", [])
        model = result.get("model", "")
        input_tokens = result.get("input_tokens", "")
        output_tokens = result.get("output_tokens", "")
        raw_response = result.get("raw_response", "")

        triplets = validate_triplets(post_id, triplets)

        elapsed = time.perf_counter() - start_time

        append_experiment_log_csv(
            experiment_csv,
            {
                "post_id": post_id,
                "strategy": strategy,
                "model": model,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "execution_time_seconds": f"{elapsed:.3f}",
                "status": "success",
            },
        )

        append_text_log(
            text_log,
            f"[SUCCESS] post_id={post_id}, strategy={strategy}, "
            f"triplets={len(triplets)}, time={elapsed:.3f}s"
        )

        return triplets

    except Exception as e:
        elapsed = time.perf_counter() - start_time

        raw_response = ""
        if hasattr(e, "raw_response"):
            raw_response = str(e.raw_response)
        else:
            raw_response = str(e)

        print(f"[FAILED] post_id={post_id}, strategy={strategy}, error_type={type(e).__name__}")
        print(f"[FAILED] error detail: {raw_response}")

        append_experiment_log_csv(
            experiment_csv,
            {
                "post_id": post_id,
                "strategy": strategy,
                "model": "",
                "input_tokens": "",
                "output_tokens": "",
                "execution_time_seconds": f"{elapsed:.3f}",
                "status": "failed",
            },
        )

        append_error_log_csv(
            error_csv,
            {
                "post_id": post_id,
                "error_type": type(e).__name__,
                "raw_response": raw_response,
            },
        )

        append_text_log(
            text_log,
            f"[FAILED] post_id={post_id}, strategy={strategy}, "
            f"error_type={type(e).__name__}, time={elapsed:.3f}s"
        )

        return []


def main():
    base_dir = Path(__file__).resolve().parent
    project_root = base_dir.parent

    input_file = project_root / "data" / "output" / "cleaned_posts_evaluated.jsonl"

    zeroshot_output = base_dir / "graph_zeroshot.json"
    fewshot_output = base_dir / "graph_fewshot.json"

    experiment_csv = project_root / "pipeline_experiment_logs.csv"
    error_csv = project_root / "pipeline_error_logs.csv"
    text_log = project_root / "pipeline_run_log.txt"

    append_text_log(str(text_log), "=" * 80)
    append_text_log(
        str(text_log),
        f"Pipeline started at {datetime.now().isoformat(timespec='seconds')}"
    )
    append_text_log(str(text_log), f"Reading input from: {input_file}")

    records = read_jsonl(str(input_file))

    zeroshot_all_triplets = []
    fewshot_all_triplets = []

    for item in records:
        post_id = item.get("post_id", "")
        text = item.get("decontextualized_text", "")

        if not post_id or not text:
            append_text_log(str(text_log), f"[SKIPPED] invalid record: {item}")
            continue

        # zeroshot
        z_triplets = process_one_record(
            post_id=post_id,
            text=text,
            strategy="zeroshot",
            experiment_csv=str(experiment_csv),
            error_csv=str(error_csv),
            text_log=str(text_log),
        )
        zeroshot_all_triplets.extend(z_triplets)

        # fewshot
        f_triplets = process_one_record(
            post_id=post_id,
            text=text,
            strategy="fewshot",
            experiment_csv=str(experiment_csv),
            error_csv=str(error_csv),
            text_log=str(text_log),
        )
        fewshot_all_triplets.extend(f_triplets)

    save_graph(zeroshot_all_triplets, str(zeroshot_output))
    save_graph(fewshot_all_triplets, str(fewshot_output))

    append_text_log(str(text_log), f"[DONE] Saved zeroshot graph to: {zeroshot_output}")
    append_text_log(str(text_log), f"[DONE] Saved fewshot graph to: {fewshot_output}")
    append_text_log(str(text_log), "Pipeline finished.")

    print(f"[Done] Zeroshot graph saved to: {zeroshot_output}")
    print(f"[Done] Fewshot graph saved to: {fewshot_output}")
    print(f"[Done] Experiment CSV saved to: {experiment_csv}")
    print(f"[Done] Error CSV saved to: {error_csv}")
    print(f"[Done] Text log saved to: {text_log}")


if __name__ == "__main__":
    main()