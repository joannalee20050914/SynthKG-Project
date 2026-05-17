import os
import time
import logging
import subprocess
import json
from pathlib import Path
from datetime import timedelta
from concurrent.futures import ThreadPoolExecutor
import time
import random

try:
    import mlx_whisper
except ImportError:
    print("錯誤：找不到 mlx_whisper 模組。請先執行：pip install mlx-whisper")
    exit(1)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# 健身專用提示詞，提供給 Whisper 參考以提高專有名詞準確率
FITNESS_PROMPT = (
    "This is a video about fitness coaching and nutrition. Please transcribe verbatim. "
    "Relevant keywords: Squat, Bench Press, Deadlift, Quads, Glutes, Core, Whey Protein, "
    "Carbohydrates, Bulking, Cutting, Calories, TDEE, RM, Training Volume, Free Weights, "
    "Machines, Mind-Muscle Connection, Hypertrophy, Macros, Set, Rep."
)

class IGReelsTranscriber:
    def __init__(self):
        self.download_dir = Path("Downloaded_Audio")
        self.output_dir = Path("output")
        self.download_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(exist_ok=True)

    def get_video_info(self, url):
        """取得 IG 影片 ID 與中介資料"""
        try:
            info_cmd = ['yt-dlp', '-j', url]
            info_result = subprocess.run(info_cmd, capture_output=True, text=True, check=True)
            info = json.loads(info_result.stdout)
            
            video_id = info.get('id', 'Unknown_ID')
            title = info.get('title', video_id)
            uploader = info.get('uploader', 'Unknown_Creator')
            # IG 的文字與 tag 通常存在 description 欄位
            description = info.get('description', '') 
            
            return video_id, title, uploader, description
        except subprocess.CalledProcessError:
            logger.error(f"yt-dlp 獲取影片資訊失敗，請確認網址是否公開: {url}")
            return None, None, None, None

    def download_audio(self, url, video_id):
        """下載影片音檔"""
        try:
            expected_filepath = Path(self.download_dir) / f"{video_id}.m4a"
            if expected_filepath.exists():
                logger.info("音檔已存在，跳過下載")
                return expected_filepath

            out_tmpl = os.path.join(self.download_dir, f"{video_id}.%(ext)s")
            download_cmd = [
                'yt-dlp', 
                '--extract-audio', 
                '--audio-format', 'm4a',
                '--audio-quality', '0', 
                '-o', out_tmpl, 
                url
            ]
            subprocess.run(download_cmd, check=True, capture_output=True)
            
            if expected_filepath.exists():
                return expected_filepath
            return None
        except Exception as e:
            logger.error(f"下載音檔發生錯誤: {e}")
            return None

    def _format_timestamp(self, seconds):
        td = timedelta(seconds=int(seconds))
        ts = str(td)
        return f"[{ts[2:]}]" if ts.startswith("0:") else f"[{ts}]"

    def transcribe_with_whisper(self, audio_path):
        """使用 mlx_whisper 進行語音轉文字"""
        logger.info("開始執行 Whisper 轉錄...")
        try:
            result = mlx_whisper.transcribe(
                str(audio_path), 
                path_or_hf_repo="mlx-community/whisper-large-v3-mlx",
                language="en", 
                initial_prompt=FITNESS_PROMPT, 
                temperature=0.0,
                condition_on_previous_text=False
            )
            lines = [
                f"{self._format_timestamp(seg['start'])} {seg['text'].strip()}"
                for seg in result.get("segments", [])
            ]
            return "\n".join(lines)
        except Exception as e:
            logger.error(f"轉錄失敗: {e}")
            return None

    def save_result(self, title, video_id, uploader, description, text):
        """儲存結果至 txt 檔案，加入 metadata，並同步輸出 JSONL"""
        if not text: return
        
        # 1. 處理檔名安全字元
        safe_title = "".join(c for c in title if c.isalnum() or c in [' ', '-', '_']).rstrip()
        if not safe_title:
            safe_title = video_id
            
        safe_uploader = "".join(c for c in uploader if c.isalnum() or c in [' ', '-', '_']).rstrip()
        if not safe_uploader:
            safe_uploader = "Unknown_Creator"
        
        clean_text = text.replace('\u2028', '\n').replace('\u2029', '\n')
          
        # 2. 儲存為人類可讀的 TXT 檔 (每支影片獨立一個檔案)
        txt_output_path = self.output_dir / f"IG_{safe_uploader}_{video_id}_raw_transcript.txt"        
        with open(txt_output_path, "w", encoding="utf-8") as f:
            f.write(f"影片標題/ID：{title}\n")
            f.write(f"創作者：{uploader}\n")
            f.write(f"原始貼文與標籤：\n{description}\n")
            f.write(f"{'=' * 50}\n\n[語音轉錄內文]\n{clean_text}\n")
        logger.info(f"TXT 轉錄結果已儲存至: {txt_output_path}")

        # 3. 儲存為機器可讀的 JSONL 檔 (所有影片統一附加到同一個檔案中)
        jsonl_output_path = self.output_dir / "raw_transcripts.jsonl"
        data_record = {
            "video_id": video_id,
            "title": title,
            "uploader": uploader,
            "description": description,
            "transcript": clean_text
        }
        # 使用 "a" (append) 模式，將新資料加到檔案最後面
        with open(jsonl_output_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(data_record, ensure_ascii=False) + "\n")
        logger.info(f"JSON 結構資料已附加至: {jsonl_output_path}")          

    def process_task(self, url, index, total):
        logger.info(f"[{index}/{total}] 開始處理網址: {url}")
        video_id, title, uploader, description = self.get_video_info(url)
        
        if not video_id:
            return url, index, "無法獲取影片資訊 (可能為私人帳號或遭到阻擋)"

        audio_path = self.download_audio(url, video_id)
        if not audio_path:
            return url, index, "下載失敗"

        transcript_text = self.transcribe_with_whisper(audio_path)
        if transcript_text:
            self.save_result(title, video_id, uploader, description, transcript_text)
            
            # 處理完畢後刪除暫存音檔
            try:
                os.remove(audio_path)
            except OSError:
                pass
            
            return url, index, "成功"
        else:
            return url, index, "轉錄失敗"

def main():
    print("===== IG Reels 語音轉錄器 (英文語系與檔案自動讀取版) =====")
    
    # 取得程式腳本所在目錄，並尋找同目錄下的 urls.txt
    script_dir = Path(__file__).parent
    url_file_path = script_dir / "urls.txt"
    
    if not url_file_path.exists():
        print(f"錯誤：找不到網址檔案 {url_file_path.absolute()}")
        print("請在同目錄下建立 urls.txt 並貼入網址。")
        return

    # 讀取檔案內容並解析網址（支援換行或空格分隔）
    with open(url_file_path, "r", encoding="utf-8") as f:
        content = f.read()
        all_urls = content.split() # split() 會自動處理空格與換行

    if not all_urls:
        return print("urls.txt 內無有效網址，程式結束。")

    print(f"\n從檔案中讀取到 {len(all_urls)} 個網址。")
    
    transcriber = IGReelsTranscriber()
    start_time = time.time()

    # 移除多執行緒，改為循序執行以確保 MLX/GPU 記憶體穩定
    for i, url in enumerate(all_urls, 1):
        url, index, status = transcriber.process_task(url, i, len(all_urls))
        print(f"完成 [{index}/{len(all_urls)}] 狀態: {status} | 網址: {url}")
        # 新增：加入 15 到 30 秒的隨機暫停，模擬人類操作節奏
        if i < len(all_urls):  # 最後一筆不需暫停
            delay = random.uniform(15, 30)
            print(f"為避免觸發 IG 防爬蟲機制，暫停 {delay:.1f} 秒...")
            time.sleep(delay)

    print(f"\n全部執行完畢，總耗時: {(time.time() - start_time) / 60:.1f} 分鐘。")


if __name__ == "__main__":
    main()