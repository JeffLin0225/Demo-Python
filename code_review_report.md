# 程式碼審查報告

> 審查範圍：`project/Python_Demo` | 日期：2026-06-19 | 檔案數：12 (Python 原始碼檔案)

## 總結

| 嚴重程度 | 數量 |
|----------|------|
| 🔴 嚴重 | 2 |
| 🟡 警告 | 3 |
| 🟢 建議 | 3 |

---

## 嚴重問題 🔴

### 1. 未處理的外部翻譯 API 異常（造成整支 API 崩潰）
- **檔案**：`Flask_BLIP.py:31` 與 `Flask_Llama_BLIP_Basic.py:42`
- **問題**：呼叫 `translator.translate(...)` 進行同步翻譯時，沒有任何 `try-except` 異常保護。
- **影響**：`googletrans` 內部使用的是 Google 翻譯的非官方免 API Key 介面，極易因為頻繁請求而被 Google 暫時封鎖 IP（HTTP 429 Too Many Requests），或是因網路逾時而拋出異常。一旦發生異常，整支 Web 服務將會中斷，並直接向用戶端回傳 HTTP 500 錯誤。
- **修復建議**：
```python
    # 建議加上例外處理與 Fallback 機制
    try:
        caption_tw = translator.translate(caption, dest="zh-tw").text
    except Exception as e:
        print(f"翻譯失敗，保留英文原意：{str(e)}")
        caption_tw = caption  # 翻譯失敗時直接回傳英文描述，避免崩潰
```

### 2. Flask 路由命名不當與回傳欄位混淆
- **檔案**：`Flask_OCR.py:76-95`
- **問題**：該 API 服務的端點定義為 `/caption`，但內部實作卻是呼叫 `PaddleOCR` 文字辨識，且 JSON 回傳的鍵值為 `"description_en"`。
- **影響**：
  1. `/caption` 路由在語意上代表「圖像描述」而非「文字辨識」，容易與 `Flask_BLIP.py` 的 `/caption` 混淆。
  2. OCR 辨識出來的往往是圖片中的中文或多國語言文字，將其命名為 `description_en`（英文描述）在邏輯上是不正確的，會嚴重誤導前端對接的開發者。
- **修復建議**：
```python
# 修改端點名稱與回傳欄位名稱
@app.route('/ocr', methods=['POST'])
def perform_ocr():
    if 'image' not in request.files:
        return jsonify({"error": "請上傳圖片"}), 400
    
    file = request.files['image']
    try:
        image = Image.open(file.stream).convert("RGB")
        image_np = np.array(image)
        success, texts = process_ocr(image_np)
    except Exception as e:
        return jsonify({"error": f"圖片處理失敗：{str(e)}"}), 400

    return jsonify({
        "success": success,
        "ocr_results": texts  # 使用更具代表性的欄位名稱
    })
```

---

## 警告 🟡

### 1. 硬編碼本機絕對路徑（跨環境部署失敗）
- **檔案**：
  - `BLIP_Model.py:17` — 硬編碼本機圖片路徑
  - `PaddleOCR_Model.py:72` — 硬編碼本機圖片路徑
  - `StableDiffusion.py:7` — 硬編碼本機模型路徑
  - `diffusionTransform.py:9` — 硬編碼本機模型路徑
- **問題**：多個檔案直接硬編碼了特定開發者的本機絕對路徑。
- **影響**：程式碼一旦移動到其他人的電腦、Docker 容器或 CI/CD 自動化測試平台，就會因為找不到該目錄或檔案而拋出 `FileNotFoundError` 錯誤。
- **修復建議**：使用相對路徑或環境變數來載入資源：
```python
import os
# 改用專案根目錄之相對路徑
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(BASE_DIR, "assets", "image.jpg")
```

### 2. Chrome 瀏覽器驅動初始化未受 `try-finally` 保護
- **檔案**：`WebScraper.py:12-14`
- **問題**：`driver = uc.Chrome(options=options)` 在 `try` 區塊外部初始化。
- **影響**：若本機未安裝 Chrome，或 Chrome 版本與驅動不相容，此行會拋出異常。由於此行在 `try` 外，`finally` 區塊將不會被執行；然而如果把該行放入 `try` 內，由於 `finally` 區塊呼叫了 `driver.quit()`，如果 driver 初始化失敗，會導致 `driver` 變數未定義而拋出 `UnboundLocalError`，掩蓋了真實的 Chrome 初始化失敗異常。
- **修復建議**：
```python
    driver = None
    try:
        options = uc.ChromeOptions()
        options.add_argument("--disable-blink-features=AutomationControlled")
        driver = uc.Chrome(options=options)
        # 進行網頁爬取...
    except Exception as e:
        print(f"搜尋過程中發生錯誤: {e}")
    finally:
        if driver is not None:
            driver.quit()
```

### 3. 硬編碼 `mps` 運算裝置限制非 Mac 環境運行
- **檔案**：`StableDiffusion.py:10`
- **問題**：直接使用 `pipe.to("mps")`，強制使用 Apple Metal 進行加速。
- **影響**：如果程式碼被部署至 NVIDIA GPU 伺服器 (CUDA) 或僅有 CPU 的 Linux 環境，程式將會崩潰。
- **修復建議**：動態偵測當前環境最適合的運算裝置：
```python
import torch

device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
pipe = pipe.to(device)
```

---

## 建議 🟢

### 1. 方法參數命名誤導
- **檔案**：`Flask_OCR.py:16`
- **問題**：`process_ocr(image_path)` 的參數名稱為 `image_path`（路徑），但實際呼叫傳入的卻是 `image_np`（Numpy 陣列）。
- **說明**：雖然 `ocr.ocr()` 可同時相容路徑字串與 Numpy 陣列，但該參數命名會讓閱讀程式碼的人產生誤解。
- **修復建議**：將參數名稱改為 `image` 或 `image_input`。

### 2. 未完全釋放 GPU 記憶體快取
- **檔案**：`BLIP_Model.py:34-35`
- **問題**：在測試腳本結束時僅使用了 `del model` 與 `del processor`，但沒有清空 PyTorch 內部的顯存快取。
- **說明**：這會使得顯示記憶體仍被 PyTorch 程序佔用，容易造成連續執行多個 AI 腳本時拋出 OOM。
- **修復建議**：參考 `StableDiffusion.py` 的做法，加入 `gc` 與 `torch` 的清理指令：
```python
import gc
import torch

del model
del processor
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
elif torch.backends.mps.is_available():
    torch.mps.empty_cache()
```

### 3. ChromaDB 向量檢索拼字錯誤
- **檔案**：`VectorDataBase/ChromaDB.py:38`
- **問題**：查詢語句硬編碼為 `query_texts=["ˇ中國在哪邊？"]`。
- **說明**：多了一個注音符號 `ˇ`，雖不影響向量庫運作，但有損程式碼整潔度。
- **修復建議**：修正為 `query_texts=["中國在哪邊？"]`。

---

## 值得肯定的地方 ✅

1. **考慮到資源管理與清理**：在 `StableDiffusion.py` 中有明確使用 `gc.collect()` 與 `torch.mps.empty_cache()` 釋放記憶體，這能大幅降低 Mac 本地端推理時系統卡頓的機率。
2. **動態 OCR 置信度過濾**：在 `PaddleOCR_Model.py` 中，巧妙地針對不同座標位置設定不同的門檻值（例如頂部跟邊緣區塊調高門檻，核心區域調低），這是一個非常務實且能有效過濾浮水印或雜訊文字的設計。
3. **優雅的分流架構**：將耗費資源的大語言模型 (Llama 3.2) 利用 Docker Ollama 獨立部署，而 Flask Web Server 僅進行輕量的任務分發與轉發，是生產環境微服務架構的良好雛形。

---

## 架構觀察

1. **缺乏組態管理系統 (Configuration Management)**：目前各 API 伺服器散落著硬編碼的伺服器埠號 (`5001`)、API 端點 (`http://localhost:11434`)。當未來服務擴充時，建議引入統一的 `config.py` 或透過 `.env` 環境變數統一管理，避免需要手動到各檔案修改設定。
2. **同步阻塞問題**：在 Flask 的路由中，所有的 AI 模型推理與外部翻譯 HTTP 請求皆是**同步阻塞**的。在高併發場景下，單個請求的延遲（例如 LLM 生成需要 5-10 秒）會導致整個伺服器無法響應其他請求。未來若要用於生產環境，應考慮改用 **FastAPI (支援 async/await)**，或是引入 **Celery 等非同步任務佇列**進行去耦合。
