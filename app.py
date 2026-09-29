"""Streamlit UI phát hiện lỗi sản phẩm bằng PatchCore.

Chạy local:  streamlit run app.py
"""

import base64
import hashlib
import io
from pathlib import Path

import pandas as pd
import streamlit as st
import torch
from PIL import Image, ImageOps

from patchcore import (
    RESULTS_PATH,
    PatchCoreModel,
    list_categories,
    load_feature_extractor,
    predict,
)
from visualization import defect_area_ratio, defect_overlay, heatmap_overlay


CATEGORY_LABELS = {
    "bottle": "Chai (bottle)",
    "cable": "Cáp điện (cable)",
    "capsule": "Viên nang (capsule)",
    "carpet": "Thảm (carpet)",
    "grid": "Lưới (grid)",
    "hazelnut": "Hạt phỉ (hazelnut)",
    "leather": "Da (leather)",
    "metal_nut": "Đai ốc kim loại (metal_nut)",
    "pill": "Viên thuốc (pill)",
    "screw": "Ốc vít (screw)",
    "tile": "Gạch (tile)",
    "toothbrush": "Bàn chải (toothbrush)",
    "transistor": "Transistor",
    "wood": "Gỗ (wood)",
    "zipper": "Khoá kéo (zipper)",
}
EXAMPLES_DIR = Path(__file__).parent / "examples"
EXAMPLE_THUMB_SIDE = 256
IMAGE_TYPES = ["png", "jpg", "jpeg", "bmp", "tif", "tiff", "webp"]
DISPLAY_MAX_SIDE = 640
MAX_CACHED_MODELS = 3

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

st.set_page_config(page_title="PatchCore – Phát hiện lỗi", page_icon="🔍", layout="wide")


# ---------------------------------------------------------------- example picker

EXAMPLE_PICKER_CSS = """
.grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(130px, 1fr));
    gap: 0.75rem;
}
.card {
    position: relative;
    padding: 0.35rem;
    border: 2px solid transparent;
    border-radius: 0.6rem;
    background: var(--st-secondary-background-color);
    cursor: pointer;
    user-select: none;
    transition: border-color 0.15s, transform 0.15s;
}
.card:hover { transform: translateY(-2px); }
.card.selected { border-color: var(--st-primary-color); }
.card img { display: block; width: 100%; aspect-ratio: 1; object-fit: cover; border-radius: 0.4rem; }
.card .label {
    margin-top: 0.3rem;
    font-size: 0.85rem;
    text-align: center;
    color: var(--st-text-color);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.card .check {
    position: absolute;
    top: 0.6rem;
    right: 0.6rem;
    width: 1.5rem;
    height: 1.5rem;
    border-radius: 50%;
    display: none;
    align-items: center;
    justify-content: center;
    background: var(--st-primary-color);
    color: white;
    font-size: 0.9rem;
    font-weight: bold;
}
.card.selected .check { display: flex; }
.actions { display: flex; gap: 0.5rem; margin-top: 0.9rem; }
button {
    padding: 0.45rem 1rem;
    border-radius: 0.5rem;
    border: 1px solid var(--st-primary-color);
    font: inherit;
    font-size: 0.95rem;
    cursor: pointer;
}
button.primary { background: var(--st-primary-color); color: white; }
button.secondary { background: transparent; color: var(--st-text-color); border-color: var(--st-border-color, #888); }
button:disabled { opacity: 0.5; cursor: not-allowed; }
"""

EXAMPLE_PICKER_JS = """
export default function(component) {
    const { data, parentElement, setTriggerValue } = component;

    // Giữ lựa chọn chưa gửi qua các lần rerun (đổi ngưỡng, ...) của cùng một instance
    const selected = parentElement.__selected ?? new Set(data.selected);
    parentElement.__selected = selected;

    parentElement.querySelector('.st-example-picker')?.remove();
    const root = document.createElement('div');
    root.className = 'st-example-picker';
    parentElement.appendChild(root);

    const grid = document.createElement('div');
    grid.className = 'grid';
    root.appendChild(grid);

    const actions = document.createElement('div');
    actions.className = 'actions';
    const submit = document.createElement('button');
    submit.className = 'primary';
    const all = document.createElement('button');
    all.className = 'secondary';
    all.textContent = 'Chọn tất cả';
    const clear = document.createElement('button');
    clear.className = 'secondary';
    clear.textContent = 'Bỏ chọn';
    actions.append(submit, all, clear);
    root.appendChild(actions);

    const cards = data.items.map((item) => {
        const card = document.createElement('div');
        card.className = 'card';
        card.title = item.label;
        card.innerHTML = '<img alt=""><div class="label"></div><div class="check">✓</div>';
        card.querySelector('img').src = item.src;
        card.querySelector('.label').textContent = item.label;
        card.onclick = () => {
            selected.has(item.id) ? selected.delete(item.id) : selected.add(item.id);
            refresh();
        };
        grid.appendChild(card);
        return [item.id, card];
    });

    function refresh() {
        cards.forEach(([id, card]) => card.classList.toggle('selected', selected.has(id)));
        submit.textContent = `🔍 Kiểm tra (${selected.size} ảnh)`;
        submit.disabled = selected.size === 0;
        clear.disabled = selected.size === 0;
    }

    submit.onclick = () => setTriggerValue('submit', [...selected]);
    all.onclick = () => { data.items.forEach((item) => selected.add(item.id)); refresh(); };
    clear.onclick = () => { selected.clear(); refresh(); };

    refresh();
}
"""

example_picker = st.components.v2.component("example_picker", css=EXAMPLE_PICKER_CSS, js=EXAMPLE_PICKER_JS)


# ---------------------------------------------------------------- model / cache

@st.cache_resource(show_spinner="Đang tải backbone WideResNet50 (lần đầu ~1 phút)...")
def get_extractor():
    return load_feature_extractor(DEVICE)


@st.cache_resource(max_entries=MAX_CACHED_MODELS, show_spinner="Đang tải memory bank...")
def get_model(category):
    return PatchCoreModel.load(category, DEVICE)


@st.cache_data
def get_results():
    return pd.read_csv(RESULTS_PATH, index_col="category")


def open_image(data):
    # exif_transpose: ảnh chụp từ điện thoại hay bị xoay theo EXIF
    return ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")


def display_size(image):
    scale = min(1.0, DISPLAY_MAX_SIDE / max(image.size))
    return round(image.width * scale), round(image.height * scale)


@st.cache_data(max_entries=128, show_spinner=False)
def run_prediction(category, image_hash, _data):
    # Cache theo (category, hash ảnh) -> đổi ngưỡng / hiển thị không phải chạy lại model
    image = open_image(_data)

    return predict(image, get_model(category), get_extractor(), DEVICE, output_size=display_size(image))


# ---------------------------------------------------------------- sidebar

def render_sidebar(categories):
    st.sidebar.header("⚙️ Cấu hình")

    category = st.sidebar.selectbox(
        "Loại sản phẩm",
        categories,
        format_func=lambda c: CATEGORY_LABELS.get(c, c),
        help="Mỗi loại sản phẩm có một mô hình riêng, train chỉ từ ảnh sản phẩm tốt.",
    )
    model = get_model(category)

    # key theo category -> đổi loại sản phẩm thì ngưỡng về giá trị mặc định của loại đó
    threshold = st.sidebar.slider(
        "Ngưỡng anomaly score",
        min_value=0.0,
        max_value=round(model.threshold * 2, 2),
        value=round(model.threshold, 3),
        step=0.005,
        format="%.3f",
        key=f"threshold_{category}",
        help=(
            f"Score ≥ ngưỡng → LỖI. Mặc định {model.threshold:.3f} là ngưỡng tối ưu F1 trên tập test. "
            "Giảm ngưỡng để bắt lỗi nhạy hơn (nhiều báo nhầm hơn), tăng để ít báo nhầm hơn."
        ),
    )

    st.sidebar.divider()
    st.sidebar.subheader("🎨 Hiển thị")

    heatmap_mode = st.sidebar.radio(
        "Thang màu heatmap",
        ["threshold", "image"],
        format_func={"threshold": "Theo ngưỡng (cố định)", "image": "Theo từng ảnh (min–max)"}.get,
        help=(
            "Theo ngưỡng: ảnh tốt trông xanh, vùng lỗi đỏ — so sánh được giữa các ảnh. "
            "Theo từng ảnh: luôn làm nổi vùng bất thường nhất, kể cả ảnh tốt."
        ),
    )
    alpha = st.sidebar.slider("Độ đậm heatmap", 0.0, 1.0, 0.5, 0.05)

    return category, threshold, heatmap_mode, alpha


# ---------------------------------------------------------------- inspect tab

def list_examples(category):
    # "good" lên đầu, còn lại các loại lỗi theo alphabet
    return sorted((EXAMPLES_DIR / category).glob("*.jpg"), key=lambda p: (p.stem != "good", p.stem))


def example_label(path):
    return "✅ good" if path.stem == "good" else f"⚠️ {path.stem.replace('_', ' ')}"


@st.cache_data(show_spinner=False)
def example_thumbnail(path_str):
    image = Image.open(path_str)
    image.thumbnail((EXAMPLE_THUMB_SIDE, EXAMPLE_THUMB_SIDE))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=85)

    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()


def collect_examples(category):
    examples = list_examples(category)

    if not examples:
        st.info(f"Chưa có ảnh mẫu cho **{CATEGORY_LABELS.get(category, category)}**.")
        return []

    # Lưu lại lần bấm "Kiểm tra" gần nhất -> đổi ngưỡng / hiển thị (rerun) vẫn giữ kết quả
    state_key = f"examples_submitted_{category}"
    submitted = st.session_state.get(state_key, [])

    st.caption("Bấm vào ảnh để chọn / bỏ chọn, rồi bấm **Kiểm tra** (ảnh lấy từ tập test MVTec AD).")

    # Chọn ảnh chỉ đổi giao diện phía trình duyệt, không rerun app; chỉ nút Kiểm tra gửi về Python.
    # key theo category -> đổi loại sản phẩm thì các lựa chọn không lẫn sang nhau
    result = example_picker(
        data={
            "items": [
                {"id": path.stem, "label": example_label(path), "src": example_thumbnail(str(path))}
                for path in examples
            ],
            "selected": [path.stem for path in submitted],
        },
        key=f"example_picker_{category}",
        on_submit_change=lambda: None,
    )

    if result.submit is not None:
        chosen = set(result.submit)
        submitted = [path for path in examples if path.stem in chosen]
        st.session_state[state_key] = submitted

    return [(example_label(path), path.read_bytes()) for path in submitted]


def collect_inputs(category):
    source = st.segmented_control(
        "Nguồn ảnh",
        ["example", "upload", "camera"],
        format_func={"example": "🖼️ Ảnh mẫu", "upload": "📁 Tải ảnh lên", "camera": "📷 Chụp ảnh"}.get,
        default="example",
        label_visibility="collapsed",
    )

    # segmented_control trả về None nếu user bỏ chọn -> coi như ảnh mẫu
    if source in ("example", None):
        return collect_examples(category)

    if source == "camera":
        shot = st.camera_input("Chụp ảnh sản phẩm")
        files = [shot] if shot else []
    else:
        files = st.file_uploader(
            "Chọn một hoặc nhiều ảnh sản phẩm",
            type=IMAGE_TYPES,
            accept_multiple_files=True,
        )

    return [(f.name, f.getvalue()) for f in files]


def render_result(name, image, prediction, threshold, heatmap_mode, alpha):
    is_defect = prediction.score >= threshold

    with st.container(border=True):
        header, badge = st.columns([4, 1], vertical_alignment="center")
        header.markdown(f"**{name}**")

        if is_defect:
            badge.badge("LỖI", icon="⚠️", color="red")
        else:
            badge.badge("ĐẠT", icon="✅", color="green")

        col_image, col_heatmap, col_defect, col_info = st.columns([3, 3, 3, 2])

        col_image.image(image, caption="Ảnh gốc", width="stretch")
        col_heatmap.image(
            heatmap_overlay(image, prediction.anomaly_map, threshold, heatmap_mode, alpha),
            caption="Heatmap bất thường",
            width="stretch",
        )
        col_defect.image(
            defect_overlay(image, prediction.anomaly_map, threshold),
            caption="Vùng vượt ngưỡng",
            width="stretch",
        )

        col_info.metric(
            "Anomaly score",
            f"{prediction.score:.4f}",
            delta=f"{prediction.score - threshold:+.4f} so với ngưỡng",
            delta_color="inverse",
        )
        col_info.metric("Diện tích vượt ngưỡng", f"{defect_area_ratio(prediction.anomaly_map, threshold):.1%}")


def render_inspect_tab(category, threshold, heatmap_mode, alpha):
    inputs = collect_inputs(category)

    if not inputs:
        st.info(
            f"Chọn ảnh mẫu hoặc tải lên ảnh **{CATEGORY_LABELS.get(category, category)}** để kiểm tra. "
            "Có thể chọn nhiều ảnh cùng lúc."
        )
        return

    results = []
    progress = st.progress(0.0, text="Đang phân tích...") if len(inputs) > 1 else None

    for index, (name, data) in enumerate(inputs, start=1):
        try:
            image = open_image(data)
        except Exception:
            st.warning(f"Không đọc được ảnh **{name}**, bỏ qua.")
            continue

        image = image.resize(display_size(image))
        prediction = run_prediction(category, hashlib.sha1(data).hexdigest(), data)
        results.append((name, image, prediction))

        if progress:
            progress.progress(index / len(inputs), text=f"Đang phân tích {index}/{len(inputs)}...")

    if progress:
        progress.empty()

    if not results:
        return

    summary = pd.DataFrame(
        {
            "Ảnh": name,
            "Anomaly score": round(prediction.score, 4),
            "Kết luận": "LỖI" if prediction.score >= threshold else "ĐẠT",
            "Diện tích vượt ngưỡng (%)": round(defect_area_ratio(prediction.anomaly_map, threshold) * 100, 2),
        }
        for name, _, prediction in results
    )
    defect_count = int((summary["Kết luận"] == "LỖI").sum())

    col1, col2, col3, col4 = st.columns([1, 1, 1, 2], vertical_alignment="bottom")
    col1.metric("Tổng số ảnh", len(summary))
    col2.metric("✅ Đạt", len(summary) - defect_count)
    col3.metric("⚠️ Lỗi", defect_count)
    col4.download_button(
        "⬇️ Tải kết quả (CSV)",
        summary.assign(category=category, threshold=threshold).to_csv(index=False).encode("utf-8-sig"),
        file_name=f"patchcore_{category}_results.csv",
        mime="text/csv",
    )

    if len(results) > 1:
        only_defects = st.toggle("Chỉ hiện ảnh lỗi", value=False)
        st.dataframe(summary, hide_index=True, width="stretch")
    else:
        only_defects = False

    for name, image, prediction in results:
        if only_defects and prediction.score < threshold:
            continue

        render_result(name, image, prediction, threshold, heatmap_mode, alpha)


# ---------------------------------------------------------------- performance tab

def render_performance_tab(category):
    results = get_results()
    per_category = results.drop(index="mean", errors="ignore")

    st.subheader("Hiệu năng trên tập test MVTec AD")

    if "mean" in results.index:
        mean = results.loc["mean"]
        col1, col2, col3 = st.columns(3)
        col1.metric("Image AUROC trung bình", f"{mean['image_auroc']:.4f}")
        col2.metric("Pixel AUROC trung bình", f"{mean['pixel_auroc']:.4f}")
        col3.metric("F1 trung bình", f"{mean['best_f1']:.4f}")

    st.bar_chart(
        per_category[["image_auroc", "pixel_auroc"]].rename(
            columns={"image_auroc": "Image AUROC", "pixel_auroc": "Pixel AUROC"}
        ),
        stack=False,
        y_label="AUROC",
    )

    table = per_category.copy()
    table.insert(0, "Loại sản phẩm", [CATEGORY_LABELS.get(c, c) for c in table.index])
    table.insert(0, "", ["👉" if c == category else "" for c in table.index])

    st.dataframe(
        table,
        hide_index=True,
        column_config={
            "image_auroc": st.column_config.NumberColumn("Image AUROC", format="%.4f"),
            "pixel_auroc": st.column_config.NumberColumn("Pixel AUROC", format="%.4f"),
            "best_threshold": st.column_config.NumberColumn("Ngưỡng", format="%.4f"),
            "best_f1": st.column_config.NumberColumn("F1", format="%.4f"),
        },
        width="stretch",
    )


# ---------------------------------------------------------------- guide tab

def render_guide_tab():
    st.markdown(
        """
### Cách dùng
1. Chọn **loại sản phẩm** ở thanh bên trái — phải khớp với sản phẩm trong ảnh.
2. Dùng **ảnh mẫu** có sẵn (1 ảnh tốt + 1 ảnh cho mỗi loại lỗi), hoặc tải lên một hoặc nhiều ảnh
   (hoặc chụp trực tiếp bằng camera).
3. Xem kết luận **ĐẠT / LỖI**, heatmap và vùng nghi lỗi. Tải kết quả về dạng CSV nếu cần.

### Đọc kết quả
- **Anomaly score**: khoảng cách từ patch bất thường nhất của ảnh tới patch "bình thường" gần nhất
  trong memory bank. Score càng cao càng bất thường.
- **Ngưỡng**: score ≥ ngưỡng → LỖI. Mặc định là ngưỡng tối ưu F1 trên tập test; có thể chỉnh ở thanh bên.
- **Heatmap**: đỏ = vùng khác biệt so với ảnh sản phẩm tốt lúc train.
- **Vùng vượt ngưỡng**: các pixel có anomaly score ≥ ngưỡng, được tô đỏ và viền.

### Lưu ý
- Mô hình chỉ học từ ảnh sản phẩm **tốt** của đúng loại đó — ảnh của vật khác sẽ gần như luôn bị báo LỖI.
- Kết quả tốt nhất khi ảnh chụp giống điều kiện dataset MVTec AD: nền, góc chụp, ánh sáng ổn định,
  sản phẩm nằm giữa khung hình.

### Mô hình
PatchCore với backbone WideResNet50-2 (feature layer2 + layer3, ảnh 224×224),
memory bank 5000 patch chọn bằng k-center greedy coreset.
"""
    )


# ---------------------------------------------------------------- main

def main():
    categories = list_categories()

    if not categories:
        st.error("Không tìm thấy checkpoint nào trong `notebooks/checkpoints/`.")
        st.stop()

    get_extractor()
    category, threshold, heatmap_mode, alpha = render_sidebar(categories)

    st.title("🔍 Phát hiện lỗi sản phẩm với PatchCore")
    st.caption(f"Đang dùng mô hình: **{CATEGORY_LABELS.get(category, category)}** · Thiết bị: `{DEVICE.type}`")

    tab_inspect, tab_performance, tab_guide = st.tabs(["🔍 Kiểm tra ảnh", "📊 Hiệu năng mô hình", "ℹ️ Hướng dẫn"])

    with tab_inspect:
        render_inspect_tab(category, threshold, heatmap_mode, alpha)

    with tab_performance:
        render_performance_tab(category)

    with tab_guide:
        render_guide_tab()


main()
