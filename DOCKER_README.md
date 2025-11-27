# SAM3 Docker + Gradio 使用指南

## 系統需求

- **GPU**: NVIDIA GPU with CUDA 12.6+ support (推薦至少 16GB VRAM)
- **Docker**: Docker 24.0+ with Docker Compose v2
- **NVIDIA Container Toolkit**: 用於 GPU 支援

## 快速開始

### 1. 設定 Hugging Face Token

SAM3 模型需要從 Hugging Face 下載。請先：

1. 前往 [Hugging Face Settings](https://huggingface.co/settings/tokens) 建立 Token
2. 前往 [SAM3 Model Page](https://huggingface.co/facebook/sam3) 申請存取權限
3. 複製 `.env.example` 為 `.env` 並填入您的 Token：

```bash
cp .env.example .env
# 編輯 .env 檔案，填入您的 HF_TOKEN
```

### 2. 啟動服務

```bash
# 使用啟動腳本
./start.sh

# 或直接使用 Docker Compose
docker compose up --build
```

### 3. 存取介面

開啟瀏覽器訪問：http://localhost:7860

## 使用方式

### 圖片分割

1. 點擊 "Image Segmentation" 標籤
2. 上傳 JPG/PNG 圖片
3. 輸入文字提示詞（例如："a person", "the dog", "red car"）
4. 調整信心閾值（預設 0.5）
5. 點擊 "Segment Image"

### 視頻分割

1. 點擊 "Video Segmentation" 標籤
2. 上傳 MP4 視頻
3. 輸入文字提示詞
4. 點擊 "Process Video"
5. 等待處理完成（視視頻長度而定）

## 提示詞範例

- `"a person wearing red"` - 穿紅色衣服的人
- `"the dog"` - 狗
- `"cars on the road"` - 路上的車
- `"a cup on the table"` - 桌上的杯子
- `"player in white jersey"` - 穿白色球衣的球員

## 停止服務

```bash
./stop.sh
# 或
docker compose down
```

## 故障排除

### GPU 無法識別

確認 NVIDIA Container Toolkit 已正確安裝：

```bash
# 測試 GPU 存取
docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu22.04 nvidia-smi
```

### 記憶體不足

- 減少視頻解析度
- 使用較短的視頻
- 增加系統 swap 空間

### 模型下載失敗

- 確認 HF_TOKEN 設定正確
- 確認已申請 SAM3 模型存取權限
- 檢查網路連線

## 目錄結構

```
sam3/
├── app.py              # Gradio 應用程式
├── Dockerfile          # Docker 映像檔定義
├── docker-compose.yml  # Docker Compose 配置
├── .env.example        # 環境變數範例
├── start.sh            # 啟動腳本
├── stop.sh             # 停止腳本
├── uploads/            # 上傳檔案目錄
└── outputs/            # 輸出檔案目錄
```

## 進階配置

### 自訂 Port

編輯 `docker-compose.yml` 或設定環境變數：

```bash
export GRADIO_SERVER_PORT=8080
docker compose up
```

### 多 GPU 支援

預設使用所有可用 GPU。若要限制使用特定 GPU：

```bash
# 只使用 GPU 0
NVIDIA_VISIBLE_DEVICES=0 docker compose up
```

## 授權

SAM3 採用 SAM License。詳見 [LICENSE](LICENSE) 檔案。
