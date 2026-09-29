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
