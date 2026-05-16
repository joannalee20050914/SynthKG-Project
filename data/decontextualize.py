import os
import json
import csv
import time
from datetime import datetime
from groq import Groq
from dotenv import load_dotenv

# 系統參數與路徑設定 (確保這幾行存在於函式之外)
INPUT_JSONL = "output/raw_transcripts.jsonl"
OUTPUT_JSONL = "output/cleaned_posts.jsonl"
ERROR_CSV = "output/error_logs.csv"


# 1. 載入 .env 檔案中的變數
load_dotenv()

# 2. 讀取金鑰
api_key = os.getenv("GROQ_API_KEY")

# 3. 檢查金鑰是否存在，若不存在則拋出錯誤並停止程式 (防呆機制)
if not api_key:
    raise ValueError("找不到 GROQ_API_KEY。請確認專案目錄下已建立 .env 檔案並設定金鑰。")

# 4. 初始化 API 客戶端
client = Groq(api_key=api_key)
MODEL_NAME = "llama-3.3-70b-versatile"

def format_prompt(metadata_title, metadata_uploader, metadata_desc, transcript):
    """
    嚴格的去脈絡化 Prompt，針對社群短影音特性最佳化。
    """
    prompt = f"""You are an expert technical editor. Your task is to rewrite the provided social media transcript into a clear, third-person educational text suitable for knowledge graph extraction.
Use the provided metadata (Title, Uploader: {metadata_uploader}, Description) for context.

STRICT RULES (YOU MUST FOLLOW THESE EXACTLY):

1. PERSPECTIVE & PRONOUNS:
   - First-Person (I, me, my, we, our): Replace with the Uploader's name ({metadata_uploader}).
   - Second-Person (you, your): Replace with generic terms like "the viewer", "a person", or rephrase into passive voice/imperative statements. NEVER replace "you" with the Uploader's name.

2. REMOVE SOCIAL MEDIA JUNK (CTA):
   - Completely DELETE any sentences asking for likes, subscribes, follows, comments, sending links, or checking descriptions.

3. HANDLE VISUAL CUES SAFELY:
   - The transcript contains terms like "like this", "here", "this position" that refer to video actions.
   - You CANNOT see the video. Do NOT guess or invent specific anatomical positions to replace them.
   - If a sentence is entirely useless without the visual context, delete it. If it contains other useful information, keep the text factual and objective without hallucinating.

4. ABSOLUTE NEGATIVE CONSTRAINTS (ANTI-HALLUCINATION & ANTI-NOTE):
   - DO NOT invent any information, numbers, products, or apps (e.g., do NOT mention any mobile apps or specific app names unless explicitly stated in the transcript).
   - DO NOT mention or cite external videos, YouTube titles, or other creators not present in the transcript.
   - DO NOT add any "Note:", explanatory text, or metadata summaries at the beginning or end of your response. 
   - Start your response directly with the first sentence of the rewritten educational text.
   
Metadata Context:
Title: {metadata_title}
Uploader: {metadata_uploader}
Description and Tags: {metadata_desc}

Transcript to Rewrite:
{transcript}
"""
    return prompt

def call_llm_api(prompt):
    """呼叫 Groq API 進行推論，設定低 Temperature 以降低幻覺風險"""
    response = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model=MODEL_NAME,
        temperature=0.1, # 降低隨機性，確保輸出一致性
        max_tokens=2048,
    )
    return response.choices[0].message.content.strip()

def process_decontextualization():
    """主要處理管線"""
    print("===== 開始執行文本去脈絡化 (Decontextualization) =====")
    
    # 確保輸出目錄存在
    os.makedirs(os.path.dirname(OUTPUT_JSONL) if os.path.dirname(OUTPUT_JSONL) else '.', exist_ok=True)
    
    # 新增修正：每次執行前若存在舊的輸出檔則直接刪除，避免新舊資料重複附加
    if os.path.exists(OUTPUT_JSONL):
        os.remove(OUTPUT_JSONL)
        print(f"已清理舊有的輸出檔案: {OUTPUT_JSONL}")

    # 初始化錯誤日誌標題 (若檔案不存在)
    if not os.path.exists(ERROR_CSV):
        with open(ERROR_CSV, 'w', encoding='utf-8', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Timestamp", "Video_ID", "Error_Type", "Error_Message"])

    processed_count = 0
    error_count = 0
    seen_ids = set()

    with open(INPUT_JSONL, 'r', encoding='utf-8') as infile:
        for line in infile:
            if not line.strip(): continue
            data = json.loads(line)
            video_id = data.get("video_id", "Unknown")

            if video_id in seen_ids:
                print(f"跳過重複項目: {video_id}")
                continue
            seen_ids.add(video_id)

            print(f"處理中: {video_id}...")
            
            try:
                # 步驟 1: 組合 Prompt
                prompt = format_prompt(
                    metadata_title=data.get("title", ""),
                    metadata_uploader=data.get("uploader", ""),
                    metadata_desc=data.get("description", ""),
                    transcript=data.get("transcript", "")
                )
                
                # 步驟 2: 呼叫 API
                decontextualized_text = call_llm_api(prompt)
                
                if not decontextualized_text:
                    raise ValueError("LLM API 回傳空值")
                
                # 步驟 3: 結構化輸出至 cleaned_posts.jsonl (保留實驗對照組)
                output_record = {
                    "post_id": video_id,
                    "original_text": data.get("transcript", ""),
                    "decontextualized_text": decontextualized_text
                }
                
                with open(OUTPUT_JSONL, 'a', encoding='utf-8') as outfile:
                    outfile.write(json.dumps(output_record, ensure_ascii=False) + "\n")
                
                processed_count += 1
                print("等待 20 秒以避免觸發 Rate Limit...")
                time.sleep(20)
                
            except Exception as e:
                # 步驟 4: 觸發防呆機制，將錯誤寫入 CSV，程式繼續執行下一筆
                print(f"錯誤: {video_id} 處理失敗，已記錄至錯誤日誌。")
                print(f"具體錯誤原因: {str(e)}")
                with open(ERROR_CSV, 'a', encoding='utf-8', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow([
                        datetime.now().isoformat(), 
                        video_id, 
                        type(e).__name__, 
                        str(e)
                    ])
                error_count += 1

    print(f"===== 執行完畢 =====")
    print(f"成功處理: {processed_count} 筆")
    print(f"失敗紀錄: {error_count} 筆 (請查看 {ERROR_CSV})")

if __name__ == "__main__":
    process_decontextualization()