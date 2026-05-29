from typing import Any, Dict, Literal

from extraction.zero_shot import (
    MODEL_NAME as ZERO_SHOT_MODEL_NAME,
    format_zero_shot_prompt,
    get_client as get_zero_shot_client,
    validate_triplets as validate_zero_shot_triplets,
)

from extraction.few_shot import (
    MODEL_NAME as FEW_SHOT_MODEL_NAME,
    format_few_shot_prompt,
    get_client as get_few_shot_client,
    validate_triplets as validate_few_shot_triplets,
)

class ExtractionError(Exception):
    """
    Extraction module custom exception.

    用途：
        當 API 呼叫、JSON 解析、Pydantic 驗證失敗時，
        把 raw_response 一起包進 exception，方便 pipeline 寫入 error_logs.csv。
    """

    def __init__(self, message: str, raw_response: str = "", original_error: Exception | None = None):
        super().__init__(message)
        self.raw_response = raw_response
        self.original_error = original_error

def _safe_get_usage_value(usage: Any, key: str) -> int:
    """
    安全取得 Groq response usage 裡的 token 數。
    不同 API 版本的 usage 可能是 dict，也可能是物件屬性。
    """

    if usage is None:
        return 0

    if isinstance(usage, dict):
        return usage.get(key, 0) or 0

    return getattr(usage, key, 0) or 0


def _call_llm_with_usage(prompt: str, strategy: Literal["zeroshot", "fewshot"]) -> Dict[str, Any]:
    """
    呼叫 Groq API，並同時回傳：
    - raw_response
    - input_tokens
    - output_tokens
    - model
    """

    if strategy == "zeroshot":
        client = get_zero_shot_client()
        model_name = ZERO_SHOT_MODEL_NAME

    elif strategy == "fewshot":
        client = get_few_shot_client()
        model_name = FEW_SHOT_MODEL_NAME

    else:
        raise ValueError("strategy must be either 'zeroshot' or 'fewshot'.")

    response = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model=model_name,
        temperature=0.1,
        max_tokens=2048,
    )

    raw_response = response.choices[0].message.content.strip()

    usage = getattr(response, "usage", None)

    input_tokens = _safe_get_usage_value(usage, "prompt_tokens")
    output_tokens = _safe_get_usage_value(usage, "completion_tokens")

    return {
        "raw_response": raw_response,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "model": model_name,
    }


def extract_triplets(post_id: str, text: str, strategy: str) -> dict:
    """
    Pipeline 統一呼叫入口。

    Error Handling:
        - 空文字：回傳空 triplets
        - strategy 錯誤：raise ValueError
        - API / JSON / Pydantic 錯誤：raise ExtractionError
          並盡量附上 raw_response
    """

    if strategy not in {"zeroshot", "fewshot"}:
        raise ValueError("strategy must be either 'zeroshot' or 'fewshot'.")

    if not text or not text.strip():
        return {
            "triplets": [],
            "model": ZERO_SHOT_MODEL_NAME if strategy == "zeroshot" else FEW_SHOT_MODEL_NAME,
            "input_tokens": 0,
            "output_tokens": 0,
            "raw_response": "",
        }

    raw_response = ""

    try:
        if strategy == "zeroshot":
            prompt = format_zero_shot_prompt(post_id, text)
            llm_result = _call_llm_with_usage(prompt, "zeroshot")
            raw_response = llm_result["raw_response"]
            validated_result = validate_zero_shot_triplets(raw_response)

        else:
            prompt = format_few_shot_prompt(post_id, text)
            llm_result = _call_llm_with_usage(prompt, "fewshot")
            raw_response = llm_result["raw_response"]
            validated_result = validate_few_shot_triplets(raw_response)

        triplets = []

        for triplet in validated_result.triplets:
            item = triplet.model_dump()
            item["source_post_id"] = post_id
            triplets.append(item)

        return {
            "triplets": triplets,
            "model": llm_result["model"],
            "input_tokens": llm_result["input_tokens"],
            "output_tokens": llm_result["output_tokens"],
            "raw_response": raw_response,
        }

    except Exception as e:
        raise ExtractionError(
            message=f"Triplet extraction failed for post_id={post_id}, strategy={strategy}: {str(e)}",
            raw_response=raw_response,
            original_error=e,
        ) from e