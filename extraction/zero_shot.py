import os
import json
import csv
import time
from datetime import datetime
from typing import Any, Dict

from groq import Groq
from dotenv import load_dotenv
from pydantic import ValidationError

from extraction.schema import TripletList


# ============================================================
# 系統參數與路徑設定
# 目的：
#   1. 讀取 A 模組產出的去脈絡化資料
#   2. 執行 Zero-shot Entity / Relation Extraction
#   3. 輸出標準化 triplets JSON 給下游圖譜模組使用
# ============================================================

INPUT_JSONL = "data/output/cleaned_posts_evaluated.jsonl"
OUTPUT_JSON = "output/triplets_zero_shot.json"
ERROR_CSV = "output/error_logs.csv"

MODEL_NAME = "llama-3.3-70b-versatile"
SLEEP_SECONDS = 10

# ============================================================
# API 初始化區
# ============================================================

load_dotenv()

client = None


def get_client() -> Groq:
    """
    延遲初始化 Groq client。
    只有真的要呼叫 API 時才檢查 GROQ_API_KEY。
    這樣單純 import prompt 或 schema 時不會直接報錯。
    """
    global client

    if client is not None:
        return client

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError("找不到 GROQ_API_KEY。請確認專案根目錄下已建立 .env 檔案並設定金鑰。")

    client = Groq(api_key=api_key)
    return client


# ============================================================
# Prompt 設計區
# Zero-shot：不提供範例，只給模型任務規則與輸出格式
# ============================================================

def format_zero_shot_prompt(source_post_id: str, text: str) -> str:
    """
    建立 Zero-shot 萃取 Prompt。

    目標：
        從去脈絡化後的文本中抽取知識三元組。
        每個三元組包含 head, relation, tail。
        若可以，保留支撐該三元組的 proposition。
    """

    prompt = f"""
You are an expert knowledge graph extraction system.

Your task is to extract entity-relation triplets from the following decontextualized text.

Definitions:
- Entity: a named object, concept, person, organization, product, location, exercise, body part, or method.
- Relation: a semantic link between two entities.
- Triplet: a structured relation in the form of head, relation, and tail.
- Proposition: a complete factual statement that supports the triplet.

Extraction rules:
1. Extract only factual information explicitly supported by the text.
2. Do not invent entities, relations, numbers, tools, products, anatomical names, or explanations.
3. Avoid pronouns such as "it", "this", "they", "he", "she" in head or tail.
4. Use the most informative entity name.
5. Keep relation phrases short but meaningful.
6. If the text does not contain useful factual relations, return an empty triplets list.
7. Output valid JSON only. Do not use markdown code fences. Do not add explanations.

Entity normalization rules:
8. The head and tail must be concise noun phrases.
9. The head and tail must be no more than 5 words each.
10. The head and tail must not be full clauses, complete sentences, or verb phrases.
11. The head and tail must not contain action descriptions such as "bending forward and kicking leg back" or "not attempting all exercises in one session".
12. Do not use vague generic entities such as "exercise", "exercises", "movement", "thing", "this", or "technique" unless the text gives a specific name.
13. Prefer specific entities, such as exact exercise names, anatomical structures, body parts, symptoms, methods, or tools.
14. Normalize repeated entities to one consistent name, especially singular/plural forms.

Safe compression rules:
15. You may compress long phrases syntactically, but only using words that appear in the input text.
16. Do not replace vague descriptions with professional or anatomical terms that are not explicitly mentioned in the text.
17. For example, do not convert "muscle in the front and side of the neck" into "sternocleidomastoid" unless "sternocleidomastoid" appears in the text.
18. If a phrase cannot be safely compressed into a concise noun phrase without inventing new meaning, do not extract that triplet.

Relation and proposition rules:
19. The relation may contain a verb, but head and tail should not be verb phrases.
20. If the relation is about exercise form, use concise relation phrases such as "requires", "involves", "targets", "improves", "affects", or "places stress on".
21. Do not over-interpret instructional phrases. If the text says someone "goes over" or "explains" an exercise, use relations like "explains" or "demonstrates" instead of "recommends".
22. Prefer objective exercise facts over viewer-centered wording.
23. Avoid using "the viewer" as an entity in head or tail.
24. Rewrite propositions as objective factual statements when possible; avoid starting propositions with "The viewer".

Required JSON format:
{{
  "triplets": [
    {{
      "source_post_id": "{source_post_id}",
      "head": "entity A",
      "relation": "semantic relation",
      "tail": "entity B",
      "proposition": "complete factual statement supporting this triplet"
    }}
  ]
}}

Source post ID:
{source_post_id}

Text:
{text}
"""
    return prompt.strip()


# ============================================================
# LLM 呼叫區
# ============================================================

def call_llm_api(prompt: str) -> str:
    """
    呼叫 Groq API 進行三元組萃取。
    temperature 設低一點，讓輸出格式比較穩定。
    """

    groq_client = get_client()

    response = groq_client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model=MODEL_NAME,
        temperature=0.1,
        max_tokens=2048,
    )

    return response.choices[0].message.content.strip()


# ============================================================
# JSON 清理與驗證區
# ============================================================

def extract_json_object(raw_text: str) -> Dict[str, Any]:
    """
    嘗試從模型輸出中取出 JSON object。

    理想情況下，模型會直接輸出：
        {"triplets": [...]}

    但有時候模型可能多輸出文字或 ```json code fence。
    這個函式會盡量把真正的 JSON 物件切出來。
    """

    cleaned = raw_text.strip()

    if cleaned.startswith("```"):
        cleaned = cleaned.replace("```json", "").replace("```", "").strip()

    start = cleaned.find("{")
    end = cleaned.rfind("}")

    if start == -1 or end == -1 or end <= start:
        raise ValueError("模型輸出中找不到合法 JSON object")

    json_text = cleaned[start:end + 1]
    return json.loads(json_text)


def validate_triplets(raw_response: str) -> TripletList:
    """
    將模型輸出轉成 JSON，並用 Pydantic schema 驗證格式。
    如果欄位缺失或空值，會丟出 ValidationError。
    """

    parsed_json = extract_json_object(raw_response)
    return TripletList.model_validate(parsed_json)


# ============================================================
# 錯誤紀錄區
# ============================================================

def init_error_log() -> None:
    """
    初始化錯誤日誌。
    如果 error_logs.csv 不存在，就建立標題列。
    """

    os.makedirs(os.path.dirname(ERROR_CSV), exist_ok=True)

    if not os.path.exists(ERROR_CSV):
        with open(ERROR_CSV, "w", encoding="utf-8", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                "Timestamp",
                "Source_Post_ID",
                "Error_Type",
                "Error_Message",
                "Raw_Response"
            ])


def log_error(source_post_id: str, error: Exception, raw_response: str = "") -> None:
    """
    將單筆資料的錯誤寫入 CSV。
    這樣某一筆失敗時，程式不會整個中斷。
    """

    with open(ERROR_CSV, "a", encoding="utf-8", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            datetime.now().isoformat(),
            source_post_id,
            type(error).__name__,
            str(error),
            raw_response
        ])


# ============================================================
# 主處理管線
# ============================================================

def process_zero_shot_extraction() -> None:
    """
    Zero-shot 萃取主流程。

    流程：
        1. 逐行讀取 cleaned_posts_evaluated.jsonl
        2. 取出 decontextualized_text
        3. 呼叫 LLM 進行三元組萃取
        4. 使用 Pydantic 驗證輸出格式
        5. 統一輸出到 triplets_zero_shot.json
    """

    print("===== 開始執行 Zero-shot Entity / Relation Extraction =====")

    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    init_error_log()

    all_triplets = []
    processed_count = 0
    error_count = 0

    with open(INPUT_JSONL, "r", encoding="utf-8") as infile:
        for line in infile:
            if not line.strip():
                continue

            data = json.loads(line)

            source_post_id = (
                data.get("post_id")
                or data.get("source_post_id")
                or data.get("video_id")
                or "Unknown"
            )

            text = data.get("decontextualized_text", "")

            if not text.strip():
                print(f"跳過空文本: {source_post_id}")
                continue

            print(f"處理中: {source_post_id}")

            raw_response = ""

            try:
                prompt = format_zero_shot_prompt(source_post_id, text)
                raw_response = call_llm_api(prompt)

                validated_result = validate_triplets(raw_response)

                for triplet in validated_result.triplets:
                    all_triplets.append(triplet.model_dump())

                processed_count += 1

                print(f"完成: {source_post_id}")
                print(f"等待 {SLEEP_SECONDS} 秒以降低 Rate Limit 風險...")
                time.sleep(SLEEP_SECONDS)

            except (json.JSONDecodeError, ValidationError, ValueError, Exception) as e:
                print(f"錯誤: {source_post_id} 萃取失敗，已記錄至錯誤日誌。")
                print(f"錯誤原因: {str(e)}")
                log_error(source_post_id, e, raw_response)
                error_count += 1

    output_data = {
        "triplets": all_triplets
    }

    with open(OUTPUT_JSON, "w", encoding="utf-8") as outfile:
        json.dump(output_data, outfile, ensure_ascii=False, indent=2)

    print("===== Zero-shot 萃取執行完畢 =====")
    print(f"成功處理: {processed_count} 筆")
    print(f"失敗紀錄: {error_count} 筆")
    print(f"輸出檔案: {OUTPUT_JSON}")


if __name__ == "__main__":
    process_zero_shot_extraction()