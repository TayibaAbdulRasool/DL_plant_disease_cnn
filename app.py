"""
Leaf Scan — a simple, single-screen UI for the Plant Disease CNN.

Run with:
    streamlit run app.py

Place your trained weights file (plant_disease_cnn.pth, saved via
torch.save(model.state_dict(), ...) in the notebook) in the same folder as
this file, or point to it in the "Model file" box in the sidebar. Without a
weights file the app runs in demo mode so you can still preview the layout.
"""

import hashlib
import io

import numpy as np
import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from model import (
    CLASS_NAMES,
    IMG_SIZE,
    NORM_MEAN,
    NORM_STD,
    load_model,
    parse_class_name,
)


def html_block(s: str) -> str:
    """Render a multi-line HTML string safely with st.markdown.

    Streamlit's markdown parser treats raw HTML as a CommonMark HTML block,
    which ends at the first blank line — anything after that gets parsed as
    a new (indented) block and shown as literal text instead of HTML. This
    strips blank lines so multi-line f-strings can stay readable in the
    source without breaking at render time.
    """
    lines = [ln for ln in s.strip("\n").splitlines() if ln.strip() != ""]
    st.markdown("\n".join(lines), unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Leaf Scan",
    page_icon="🌿",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
st.markdown(
    """
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700;800&family=Dancing+Script:wght@600;700&display=swap" rel="stylesheet">

    <style>
    :root{
        --bg-top: #6F9078;
        --bg-mid: #5AA274;
        --bg-low: #1B5850;
        --bg-bottom: #04211F;
        --card: rgba(18, 46, 43, 0.55);
        --card-border: rgba(223, 232, 170, 0.22);
        --accent: #DDE59B;
        --accent-soft: rgba(221, 229, 155, 0.16);
        --accent-strong: #A9E6B4;
        --warn: #F0A75C;
        --warn-soft: rgba(240, 167, 92, 0.18);
        --text: #EEF3EA;
        --text-soft: rgba(238, 243, 234, 0.66);
        --btn-dark: #0F2C29;
        --line: rgba(238, 243, 234, 0.18);
    }

    html, body, [class*="css"], .stApp{
        font-family: 'Nunito', -apple-system, sans-serif;
        color: var(--text);
    }
    .stApp{
        background: linear-gradient(180deg,
            var(--bg-top) 0%,
            var(--bg-mid) 26%,
            var(--bg-low) 62%,
            var(--bg-bottom) 100%) !important;
        background-attachment: fixed !important;
    }

    section[data-testid="stSidebar"]{
        background: var(--bg-bottom) !important;
    }
    section[data-testid="stSidebar"] *{
        color: var(--text) !important;
    }

    #MainMenu, footer, header[data-testid="stHeader"]{
        visibility: hidden;
        height: 0;
    }
    .block-container{
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 460px;
    }

    /* ---------- tiny header ---------- */
    .app-name{
        font-family: 'Dancing Script', cursive;
        font-size: 3rem;
        font-weight: 700;
        color: var(--accent);
        margin-bottom: 14px;
        text-align: center;
        text-shadow: 0 2px 10px rgba(0,0,0,0.25);
    }

    /* ---------- upload frame ---------- */
    section[data-testid="stFileUploaderDropzone"]{
        background: var(--card);
        border: 2px dashed var(--card-border);
        border-radius: 20px;
        padding: 6px;
    }
    section[data-testid="stFileUploaderDropzone"] *{
        color: var(--text) !important;
    }
    div[data-testid="stFileUploaderDropzoneInstructions"]{
        text-align: center;
        justify-content: center;
    }
    div[data-testid="stFileUploaderDropzoneInstructions"] > div{
        align-items: center;
    }
    section[data-testid="stFileUploaderDropzone"] button{
        border: 1px solid var(--accent) !important;
        background: var(--accent-soft) !important;
        color: var(--accent) !important;
        border-radius: 999px !important;
        font-weight: 700;
    }
    div[data-testid="stFileUploaderDropzoneInstructions"] span,
    div[data-testid="stFileUploaderDropzoneInstructions"] small{
        color: var(--text-soft) !important;
        text-align: center;
    }
    section[data-testid="stFileUploaderDropzone"] svg{
        fill: var(--accent) !important;
    }

    img{
        border-radius: 18px;
    }

    /* ---------- buttons ---------- */
    .stButton > button, .stDownloadButton > button{
        background: var(--btn-dark) !important;
        color: var(--text) !important;
        border: 1px solid rgba(238,243,234,0.12) !important;
        border-radius: 999px !important;
        font-weight: 800;
        font-size: 0.95rem;
        letter-spacing: 0.04em;
        padding: 0.75rem 1.4rem;
        font-family: 'Nunito', sans-serif;
        box-shadow: 0 6px 18px rgba(0,0,0,0.35);
    }
    .stButton > button:hover, .stDownloadButton > button:hover{
        background: #163934 !important;
        border-color: var(--accent) !important;
    }

    /* ---------- result card ---------- */
    .result-card{
        background: var(--card);
        border: 1px solid var(--card-border);
        backdrop-filter: blur(6px);
        border-radius: 20px;
        padding: 32px 24px 26px 24px;
        text-align: center;
        margin-top: 18px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
    }
    .result-emoji{ font-size: 3rem; line-height: 1; margin-bottom: 14px; }
    .status-circle{
        width: 64px;
        height: 64px;
        border-radius: 50%;
        margin: 0 auto 16px auto;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .status-circle.ok{ background: var(--accent-soft); }
    .status-circle.warn{ background: var(--warn-soft); }
    .status-badge{
        display: inline-block;
        font-size: 0.72rem;
        font-weight: 800;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        padding: 4px 12px;
        border-radius: 999px;
        margin-bottom: 10px;
    }
    .status-badge.ok{ background: var(--accent-soft); color: var(--accent-strong); }
    .status-badge.warn{ background: var(--warn-soft); color: var(--warn); }
    .status-title{
        font-size: 1.25rem;
        font-weight: 800;
        color: var(--text);
        margin: 0;
    }
    .status-disease{
        font-size: 1rem;
        font-weight: 700;
        color: var(--accent);
        margin-top: 4px;
    }
    .status-note{
        font-size: 0.9rem;
        font-weight: 600;
        color: var(--text-soft);
        margin-top: 4px;
    }
    .status-sub{
        font-size: 0.85rem;
        color: var(--text-soft);
        margin-top: 4px;
    }

    /* ---------- banner ---------- */
    .demo-banner{
        background: var(--warn-soft);
        border: 1px solid var(--warn);
        color: var(--text);
        font-size: 0.82rem;
        padding: 10px 14px;
        border-radius: 14px;
        margin-bottom: 16px;
    }
    .demo-banner code{
        background: rgba(0,0,0,0.25);
        color: var(--accent);
        padding: 1px 5px;
        border-radius: 4px;
    }
    </style>

    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Model loading (cached)
# ---------------------------------------------------------------------------
DEFAULT_WEIGHTS_PATH = "plant_disease_cnn.pth"


@st.cache_resource(show_spinner=False)
def get_model(weights_path):
    try:
        return load_model(weights_path, device="cpu"), None
    except Exception as exc:  # file missing, corrupt, shape mismatch, etc.
        return None, str(exc)


def preprocess(image: Image.Image) -> torch.Tensor:
    tfm = transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=NORM_MEAN, std=NORM_STD),
        ]
    )
    return tfm(image.convert("RGB")).unsqueeze(0)


def demo_prediction(image_bytes: bytes):
    """Deterministic pseudo-prediction used only when no trained weights are
    available, so the interface can still be previewed end to end."""
    seed = int(hashlib.sha256(image_bytes).hexdigest(), 16) % (2**32)
    rng = np.random.default_rng(seed)
    logits = rng.normal(loc=0, scale=1.4, size=len(CLASS_NAMES))
    top = rng.integers(0, len(CLASS_NAMES))
    logits[top] += 4.5
    probs = np.exp(logits) / np.exp(logits).sum()
    return probs


def run_inference(model, image: Image.Image):
    tensor = preprocess(image)
    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1).squeeze(0).numpy()
    return probs


with st.sidebar:
    st.markdown("**Model file**")
    weights_path = st.text_input(
        "Path to .pth weights", value=DEFAULT_WEIGHTS_PATH, label_visibility="collapsed"
    )
    st.caption(
        "Defaults to `plant_disease_cnn.pth` next to app.py — the file saved "
        "by the notebook's `torch.save(model.state_dict(), ...)` step."
    )

model, load_error = get_model(weights_path)
if load_error and model is None:
    with st.sidebar:
        st.caption(f"Could not load weights from `{weights_path}`: {load_error}")

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown('<div class="app-name">🌿 Leaf Scan</div>', unsafe_allow_html=True)

if model is None:
    html_block(
        f"""<div class="demo-banner">
        Demo mode — no weights found at <code>{weights_path}</code>. Results
        below are simulated so you can preview the app.
        </div>"""
    )

# ---------------------------------------------------------------------------
# Reset support: bump the uploader's key to clear it on "Scan again"
# ---------------------------------------------------------------------------
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

uploaded = st.file_uploader(
    "Upload a leaf photo",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed",
    key=f"uploader_{st.session_state.uploader_key}",
)

if uploaded is not None:
    image = Image.open(io.BytesIO(uploaded.getvalue()))
    st.image(image, use_container_width=True)

    if model is not None:
        probs = run_inference(model, image)
    else:
        probs = demo_prediction(uploaded.getvalue())

    top_idx = int(np.argmax(probs))
    top_name = CLASS_NAMES[top_idx]
    crop, condition, is_healthy = parse_class_name(top_name)
    confidence = probs[top_idx] * 100

    if is_healthy:
        emoji, circle_class = "😊", "ok"
        badge_class, badge_text = "ok", "Healthy"
        title = f"{crop} Leaf"
        subtitle_class = "status-note"
        subtitle = "No signs of disease detected"
        icon_svg = (
            '<svg width="28" height="28" viewBox="0 0 24 24" fill="none">'
            '<path d="M5 13l4 4L19 7" stroke="var(--accent-strong)" stroke-width="2.5" '
            'stroke-linecap="round" stroke-linejoin="round"/></svg>'
        )
    else:
        emoji, circle_class = "😕", "warn"
        badge_class, badge_text = "warn", "Disease detected"
        title = f"{crop} Leaf"
        subtitle_class = "status-disease"
        subtitle = condition
        icon_svg = (
            '<svg width="26" height="26" viewBox="0 0 24 24" fill="none">'
            '<path d="M12 8v5" stroke="var(--warn)" stroke-width="2.5" stroke-linecap="round"/>'
            '<circle cx="12" cy="16.5" r="1.3" fill="var(--warn)"/>'
            '<circle cx="12" cy="12" r="9.5" stroke="var(--warn)" stroke-width="1.6"/></svg>'
        )

    html_block(
        f"""
        <div class="result-card">
            <div class="result-emoji">{emoji}</div>
            <div class="status-circle {circle_class}">{icon_svg}</div>
            <span class="status-badge {badge_class}">{badge_text}</span>
            <p class="status-title">{title}</p>
            <p class="{subtitle_class}">{subtitle}</p>
            <p class="status-sub">{confidence:.0f}% confidence</p>
        </div>
        """
    )

    st.markdown('<div style="height:16px;"></div>', unsafe_allow_html=True)
    if st.button("Scan again", use_container_width=True):
        st.session_state.uploader_key += 1
        st.rerun()
