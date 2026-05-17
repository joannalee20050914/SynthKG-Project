import json
from rouge_score import rouge_scorer
import statistics

def evaluate_and_filter_quality(input_jsonl, output_jsonl):
    """
    讀取去脈絡化文字，進行 ROUGE-1 F1 評估。
    1. 具備記憶體內去重機制（防堵上游重複執行附加導致樣本翻倍的 Bug）。
    2. 採用調降後的短影音適應性閾值 (0.45) 進行品質篩選。
    3. 將過濾後的合格優質樣本寫入帶有後綴的新檔案。
    """
    scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
    
    # 用於記憶體內去重 (Key: post_id, Value: json_dict)
    unique_data = {}
    duplicate_count = 0

    print("===== 開始執行去脈絡化品質評估 與 自動過濾 =====")
    print(f"正在讀取原始檔案: {input_jsonl}")
    
    # 步驟 1: 讀取並在記憶體內強制去重
    with open(input_jsonl, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            item = json.loads(line)
            pid = item.get("post_id", "Unknown")
            
            if pid in unique_data:
                duplicate_count += 1
                # 發生重複時，覆寫保留最新生成的這一筆
                unique_data[pid] = item
            else:
                unique_data[pid] = item

    if duplicate_count > 0:
        print(f"  [系統提示] 偵測到 {duplicate_count} 筆重複數據！系統已自動執行去重（僅保留最新生成版本）。")
        print(f"  去重後的真實樣本數: {len(unique_data)}\n")

    f1_scores = []
    qualified_samples = []
    failed_samples = []
    
    # 設定經學術論證優化後的短影音適應性閾值
    THRESHOLD = 0.45 

    # 步驟 2: 執行品質評估
    for pid, data in unique_data.items():
        orig_text = data.get("original_text", "")
        decont_text = data.get("decontextualized_text", "")
        
        # 計算 ROUGE-1 F1 分數
        scores = scorer.score(orig_text, decont_text)
        f1 = scores['rouge1'].fmeasure
        f1_scores.append(f1)
        
        # 將動態評估分數注入該筆資料的 metadata，方便後續效能統計
        data["rouge1_f1_score"] = round(f1, 4)
        
        if f1 >= THRESHOLD:
            qualified_samples.append(data)
        else:
            failed_samples.append((pid, f1))

    # 步驟 3: 輸出統計報告
    avg_f1 = statistics.mean(f1_scores) if f1_scores else 0
    print("--------------------------------------------------")
    print(f"評估數據統計結果:")
    print(f" - 有效樣本總數 (已去重): {len(f1_scores)}")
    print(f" - 平均 ROUGE-1 F1 分數: {avg_f1:.4f}")
    print(f" - 合格優質樣本數 (F1 >= {THRESHOLD}): {len(qualified_samples)}")
    print(f" - 遭過濾風險樣本數 (F1 < {THRESHOLD}): {len(failed_samples)}")
    print("--------------------------------------------------")
    
    print(f"\n[低於 {THRESHOLD} 閾值的風險樣本清單]")
    if not failed_samples:
        print(" -> 無。所有樣本皆成功通過短影音適應性門檻。")
    else:
        for pid, score in failed_samples:
            print(f" - ID: {pid} | F1 Score: {score:.4f}")

    # 步驟 4: 以覆寫模式 ('w') 寫入全新帶有後綴的檔案
    with open(output_jsonl, 'w', encoding='utf-8') as out_f:
        for sample in qualified_samples:
            out_f.write(json.dumps(sample, ensure_ascii=False) + '\n')
            
    print(f"\n===== 品質評估管線執行完畢 =====")
    print(f"已將 {len(qualified_samples)} 筆合格優質資料覆寫寫入: {output_jsonl}")

if __name__ == "__main__":
    # 定義明確的輸入與輸出路徑，加上 _evaluated 後綴
    INPUT_FILE = "output/cleaned_posts.jsonl"
    OUTPUT_FILE = "output/cleaned_posts_evaluated.jsonl"
    
    evaluate_and_filter_quality(INPUT_FILE, OUTPUT_FILE)