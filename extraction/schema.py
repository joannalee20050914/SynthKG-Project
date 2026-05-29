from typing import List, Optional
from pydantic import BaseModel, Field, field_validator


# ============================================================
# Schema 定義區
# 目的：
#   1. 規定 LLM 萃取結果的標準格式
#   2. 避免模型輸出缺少欄位或產生空值
#   3. 讓後續 Graph / RAG 模組可以穩定讀取 triplets.json
# ============================================================


class Triplet(BaseModel):
    """
    單一知識三元組資料格式。

    一筆 Triplet 代表知識圖譜中的一條邊：
        head --relation--> tail

    例如：
        深蹲 --訓練部位--> 股四頭肌
    """

    # 來源貼文 ID，用來追蹤這個三元組來自哪一筆原始資料
    source_post_id: str = Field(
        ...,
        description="The source post ID where this triplet comes from."
    )

    # 三元組的主體，也就是關係的起點 entity
    head: str = Field(
        ...,
        description="The head entity of the triplet."
    )

    # head 和 tail 之間的語意關係
    relation: str = Field(
        ...,
        description="The semantic relation between head and tail."
    )

    # 三元組的客體，也就是關係的終點 entity
    tail: str = Field(
        ...,
        description="The tail entity of the triplet."
    )

    # 支撐此三元組的完整事實敘述，可用於報告或後續檢查
    # 這個欄位不是圖譜邊的必要欄位，因此設為 Optional
    proposition: Optional[str] = Field(
        default=None,
        description="The complete factual statement supporting this triplet."
    )

    @field_validator("source_post_id", "head", "relation", "tail")
    @classmethod
    def not_empty(cls, value: str) -> str:
        """
        防呆驗證：
        確保必要欄位不能是空字串。
        如果 LLM 產生空的 head / relation / tail，就讓 Pydantic 擋下來。
        """
        if not value or not value.strip():
            raise ValueError("Field cannot be empty.")
        return value.strip()


class TripletList(BaseModel):
    """
    多筆三元組的包裝格式。

    最後輸出的 triplets.json 建議長這樣：
    {
        "triplets": [
            {
                "source_post_id": "...",
                "head": "...",
                "relation": "...",
                "tail": "...",
                "proposition": "..."
            }
        ]
    }
    """

    # 所有從文本中萃取出的三元組列表
    triplets: List[Triplet]