import os

DATA_FILE = "graph_data.json"   # 正式圖譜資料

TOP_M = 10
TOP_K = 5
N_HOP = 2

EMBED_MODEL = "all-MiniLM-L6-v2"

# reranker provider:
# "none" = 不用 LLM API
# "gemini" = 用 Gemini API
RERANK_PROVIDER = "gemini"

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = "gemini-2.5-flash"