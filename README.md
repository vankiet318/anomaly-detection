# Phát hiện lỗi sản phẩm với PatchCore

UI Streamlit cho mô hình PatchCore đã train trên 15 loại sản phẩm của MVTec AD
(notebook train: `notebooks/defect-detection.ipynb`).

## Cấu trúc

```
app.py                  # UI Streamlit
patchcore.py            # load checkpoint + inference
visualization.py        # heatmap / vùng lỗi
notebooks/checkpoints/  # <category>.pt + results.csv
examples/<category>/    # ảnh mẫu: good.jpg + <loại lỗi>.jpg (từ tập test MVTec AD, CC BY-NC-SA 4.0)
requirements.txt
.streamlit/config.toml
```

## Chạy local

```bash
python -m venv .venv
.venv\Scripts\activate          # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Lần chạy đầu sẽ tải trọng số WideResNet50-2 (~263 MB) từ `download.pytorch.org` và cache lại.

## Deploy lên Streamlit Community Cloud

1. Push repo này lên GitHub (checkpoint ~15 MB/file, dưới giới hạn 100 MB của GitHub nên không cần Git LFS).
2. Vào <https://share.streamlit.io> → **Create app** → chọn repo, branch `main`, main file `app.py`.
3. Trong **Advanced settings** chọn Python **3.12**.
4. **Deploy**. Build lần đầu mất vài phút (cài torch CPU).

`requirements.txt` dùng index CPU của PyTorch để tránh cài bản CUDA nặng vài GB.

## Thêm / cập nhật mô hình

Copy file `<category>.pt` do notebook sinh ra vào `notebooks/checkpoints/` (và cập nhật `results.csv`).
App tự nhận mọi file `.pt` trong thư mục này.
