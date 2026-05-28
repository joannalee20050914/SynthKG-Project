import os
from dotenv import load_dotenv

load_dotenv()

DEFAULT_DATA_FILE = "graph_data.json"
# 之後可以用，Ex: python graph_rag.py --graph ../extraction/outputs/graph_fewshot.json 參數設定
# Query也可以加在後面 --query "What posture cues are important in a deadlift?"

TOP_M = 10
TOP_K = 5
N_HOP = 2

EMBED_MODEL = "all-MiniLM-L6-v2"

RERANK_PROVIDER = os.getenv("RERANK_PROVIDER", "gemini")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
HF_TOKEN = os.getenv("HF_TOKEN", "")

GEMINI_MODEL = "gemini-2.5-flash"