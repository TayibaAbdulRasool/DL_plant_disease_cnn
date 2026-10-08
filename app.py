"""
The Leaf Ledger — a field-diagnostic UI for the Plant Disease CNN.

Run with:
    streamlit run app.py

Place your trained weights file (plant_disease_cnn.pth, saved via"""
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
    .status-title{
        font-size: 1.25rem;
        font-weight: 800;
        color: var(--text);
        margin: 0;
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
        emoji, circle_class, title = "😊", "ok", f"Healthy {crop} Leaf"
        icon_svg = (
            '<svg width="28" height="28" viewBox="0 0 24 24" fill="none">'
            '<path d="M5 13l4 4L19 7" stroke="var(--accent-strong)" stroke-width="2.5" '
            'stroke-linecap="round" stroke-linejoin="round"/></svg>'
        )
    else:
        emoji, circle_class, title = "😕", "warn", f"{crop} — {condition}"
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
            <p class="status-title">{title}</p>
            <p class="status-sub">{confidence:.0f}% confidence</p>
        </div>
        """
    )

    st.markdown('<div style="height:16px;"></div>', unsafe_allow_html=True)
    if st.button("Scan again", use_container_width=True):
        st.session_state.uploader_key += 1
        st.rerun()
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
    PER_CLASS_ACCURACY,
    care_notes,
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
    page_title="The Leaf Ledger",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
st.markdown(
    """
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,400;0,9..144,500;0,9..144,600;1,9..144,500&family=Public+Sans:wght@400;500;600;700&display=swap" rel="stylesheet">

    <style>
    :root{
        --paper: #EFF3EA;
        --paper-card: #FBFAF5;
        --paper-card-alt: #F4F6EE;
        --ink: #1F3D2B;
        --ink-soft: #51624F;
        --ink-faint: #8B9787;
        --line: #D7DCC9;
        --gold: #B8901F;
        --gold-soft: #E9DCB0;
        --healthy: #3F7A4E;
        --healthy-bg: #E4EFE0;
        --disease: #A63D2F;
        --disease-bg: #F4E4DD;
    }

    html, body, [class*="css"], .stApp{
        font-family: 'Public Sans', -apple-system, sans-serif;
        background-color: var(--paper) !important;
        color: var(--ink);
    }

    #MainMenu, footer, header[data-testid="stHeader"]{
        visibility: hidden;
        height: 0;
    }
    .block-container{
        padding-top: 2.2rem;
        padding-bottom: 3rem;
        max-width: 1180px;
    }

    h1, h2, h3, h4{
        font-family: 'Fraunces', Georgia, serif;
        color: var(--ink);
        letter-spacing: -0.01em;
    }

    /* ---------- masthead ---------- */
    .masthead{
        display: flex;
        justify-content: space-between;
        align-items: flex-end;
        border-bottom: 3px solid var(--ink);
        padding-bottom: 14px;
        margin-bottom: 4px;
    }
    .masthead-title{
        font-family: 'Fraunces', Georgia, serif;
        font-weight: 600;
        font-style: italic;
        font-size: 2.6rem;
        line-height: 1;
        color: var(--ink);
        margin: 0;
    }
    .masthead-sub{
        font-size: 0.95rem;
        color: var(--ink-soft);
        margin-top: 6px;
        max-width: 480px;
    }
    .masthead-leaf{ flex-shrink: 0; margin-left: 24px; color: var(--ink); }
    .masthead-rule{
        border: none;
        border-top: 1px solid var(--line);
        margin: 6px 0 28px 0;
    }

    /* ---------- tabs as pill nav ---------- */
    div[data-testid="stTabs"] button[data-baseweb="tab"]{
        font-family: 'Public Sans', sans-serif;
        font-weight: 600;
        font-size: 0.92rem;
        background: transparent;
        border: none;
        padding: 8px 4px;
        margin-right: 26px;
    }
    div[data-testid="stTabs"] button[data-baseweb="tab"] p{
        color: var(--ink-soft) !important;
        font-weight: 600;
    }
    div[data-testid="stTabs"] button[data-baseweb="tab"][aria-selected="true"] p{
        color: var(--ink) !important;
    }
    div[data-testid="stTabs"] div[data-baseweb="tab-highlight"]{
        background-color: var(--gold) !important;
        height: 3px;
    }
    div[data-testid="stTabs"] div[data-baseweb="tab-border"]{
        background-color: var(--line);
    }

    /* ---------- cards (real st.container(border=True), not hand-rolled divs) ---------- */
    div[data-testid="stVerticalBlockBorderWrapper"] > div > div[data-testid="stVerticalBlock"]{
        background: var(--paper-card);
        border-radius: 4px;
    }
    div[data-testid="stVerticalBlockBorderWrapper"]{
        border: 1px solid var(--line) !important;
        border-radius: 4px !important;
        background: var(--paper-card);
        transition: border-color 0.15s ease;
    }
    div[data-testid="stVerticalBlockBorderWrapper"]:has(section[data-testid="stFileUploaderDropzone"]):hover,
    div[data-testid="stVerticalBlockBorderWrapper"]:has(button):hover{
        border-color: var(--ink-faint) !important;
    }
    .card-label{
        font-size: 0.75rem;
        font-weight: 600;
        color: var(--gold);
        letter-spacing: 0.04em;
        margin-bottom: 10px;
        text-transform: none;
    }

    /* ---------- uploader ---------- */
    section[data-testid="stFileUploaderDropzone"]{
        background: var(--paper-card-alt);
        border: 1.5px dashed var(--ink-faint);
        border-radius: 3px;
    }
    section[data-testid="stFileUploaderDropzone"] button{
        border: 1px solid var(--ink) !important;
        background: transparent !important;
        color: var(--ink) !important;
        border-radius: 2px !important;
        font-weight: 600;
    }
    div[data-testid="stFileUploaderDropzoneInstructions"] span{
        color: var(--ink-soft);
    }

    /* ---------- buttons ---------- */
    .stButton > button, .stDownloadButton > button{
        background: var(--ink) !important;
        color: var(--paper-card) !important;
        border: none !important;
        border-radius: 2px !important;
        font-weight: 600;
        padding: 0.55rem 1.4rem;
        font-family: 'Public Sans', sans-serif;
    }
    .stButton > button:hover, .stDownloadButton > button:hover{
        background: var(--gold) !important;
        color: var(--ink) !important;
    }

    /* ---------- result header ---------- */
    .result-wrap{ position: relative; }
    .id-stamp{
        position: absolute;
        top: -6px;
        right: 0;
        width: 88px;
        height: 88px;
        border-radius: 50%;
        border: 2px solid currentColor;
        outline: 1.5px solid currentColor;
        outline-offset: 4px;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        transform: rotate(-9deg);
        font-family: 'Fraunces', Georgia, serif;
    }
    .id-stamp .num{ font-size: 1.55rem; font-weight: 600; line-height: 1; }
    .id-stamp .unit{ font-size: 0.62rem; font-family: 'Public Sans', sans-serif; margin-top: 2px; letter-spacing: 0.03em; }
    .result-eyebrow{
        font-size: 0.78rem;
        color: var(--ink-faint);
        margin-bottom: 2px;
    }
    .result-crop{
        font-family: 'Fraunces', Georgia, serif;
        font-size: 1.9rem;
        font-weight: 600;
        color: var(--ink);
        margin: 0;
        line-height: 1.15;
    }
    .result-condition{
        font-size: 1.05rem;
        color: var(--ink-soft);
        margin-top: 2px;
    }
    .status-tag{
        display: inline-block;
        font-size: 0.78rem;
        font-weight: 600;
        padding: 3px 11px;
        border-radius: 2px;
        margin-top: 10px;
    }
    .status-healthy{ background: var(--healthy-bg); color: var(--healthy); }
    .status-disease{ background: var(--disease-bg); color: var(--disease); }

    /* ---------- confidence bar ---------- */
    .conf-row{ margin-top: 20px; }
    .conf-label{
        display: flex;
        justify-content: space-between;
        font-size: 0.82rem;
        color: var(--ink-soft);
        margin-bottom: 5px;
    }
    .conf-track{
        width: 100%;
        height: 9px;
        background: var(--line);
        border-radius: 5px;
        overflow: hidden;
    }
    .conf-fill{
        height: 100%;
        background: var(--gold);
        border-radius: 5px;
    }
    .conf-fill.alt{ background: var(--ink-faint); }

    /* ---------- empty state ---------- */
    .empty-ledger{
        padding: 10px 4px;
    }
    .empty-ledger p{
        color: var(--ink-faint);
        font-size: 0.9rem;
        border-bottom: 1px solid var(--line);
        padding-bottom: 14px;
        margin-bottom: 14px;
    }

    /* ---------- species index ---------- */
    .crop-heading{
        font-family: 'Fraunces', Georgia, serif;
        font-weight: 600;
        font-size: 1.15rem;
        color: var(--ink);
        margin: 22px 0 8px 0;
        border-bottom: 1px solid var(--line);
        padding-bottom: 6px;
    }
    .species-row{
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 7px 2px;
        font-size: 0.88rem;
        border-bottom: 1px dotted var(--line);
    }
    .species-row .cond{ color: var(--ink-soft); }
    .species-row .acc{ color: var(--ink-faint); font-variant-numeric: tabular-nums; }

    /* ---------- banner ---------- */
    .demo-banner{
        background: var(--gold-soft);
        border: 1px solid var(--gold);
        color: #5C4A12;
        font-size: 0.85rem;
        padding: 10px 16px;
        border-radius: 3px;
        margin-bottom: 18px;
    }

    /* ---------- footer note ---------- */
    .caveat{
        font-size: 0.78rem;
        color: var(--ink-faint);
        margin-top: 18px;
        border-top: 1px solid var(--line);
        padding-top: 10px;
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


# ---------------------------------------------------------------------------
# Masthead
# ---------------------------------------------------------------------------
n_species = len({parse_class_name(c)[0] for c in CLASS_NAMES})

LEAF_GLYPH = (
    '<svg class="masthead-leaf" width="56" height="72" viewBox="0 0 56 72" '
    'fill="none" xmlns="http://www.w3.org/2000/svg">'
    '<path d="M28 4C10 14 4 32 8 50c3 12 12 18 20 18s17-6 20-18c4-18-2-36-20-46Z" '
    'stroke="currentColor" stroke-width="2" fill="none"/>'
    '<path d="M28 8V64" stroke="#B8901F" stroke-width="1.6"/>'
    '<path d="M28 22 18 16M28 34 15 26M28 46 16 40M28 22 38 16M28 34 41 26M28 46 40 40" '
    'stroke="currentColor" stroke-width="1.1" stroke-linecap="round"/>'
    "</svg>"
)

masthead_html = (
    '<div class="masthead">'
    "<div>"
    '<p class="masthead-title">The Leaf Ledger</p>'
    '<p class="masthead-sub">A convolutional field log for reading leaf condition across '
    f"{n_species} crop species and {len(CLASS_NAMES)} recognized conditions.</p>"
    "</div>"
    f"{LEAF_GLYPH}"
    "</div>"
    '<hr class="masthead-rule">'
)
st.markdown(masthead_html, unsafe_allow_html=True)

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
# Tabs
# ---------------------------------------------------------------------------
tab_diagnose, tab_index = st.tabs(["Diagnose", "Species index"])

# ---------------------------------------------------------------------------
# Diagnose tab
# ---------------------------------------------------------------------------
with tab_diagnose:
    if model is None:
        st.markdown(
            f"""<div class="demo-banner">
            Demo mode — no trained weights found at <code>{weights_path}</code>.
            The layout below still works end to end using a simulated result,
            so you can preview it. Add your <code>plant_disease_cnn.pth</code>
            file to enable real predictions.
            </div>""",
            unsafe_allow_html=True,
        )

    col_left, col_right = st.columns([0.42, 0.58], gap="large")

    with col_left:
        with st.container(border=True):
            st.markdown('<div class="card-label">New specimen</div>', unsafe_allow_html=True)
            uploaded = st.file_uploader(
                "Upload a leaf photo",
                type=["jpg", "jpeg", "png"],
                label_visibility="collapsed",
            )
            image = None
            if uploaded is not None:
                image = Image.open(io.BytesIO(uploaded.getvalue()))
                st.image(image, use_container_width=True)
            run = st.button(
                "Diagnose specimen", use_container_width=True, disabled=uploaded is None
            )

    with col_right:
        with st.container(border=True):
            st.markdown('<div class="card-label">Diagnosis report</div>', unsafe_allow_html=True)

            if uploaded is None or not run:
                st.markdown(
                    """
                    <div class="empty-ledger">
                    <p>Crop &amp; condition — awaiting specimen</p>
                    <p>Confidence — awaiting specimen</p>
                    <p>Care notes — awaiting specimen</p>
                    </div>
                    <p style="color:var(--ink-faint); font-size:0.85rem;">
                    Upload a clear, well-lit photo of a single leaf, then press
                    "Diagnose specimen" to fill in this report.
                    </p>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                if model is not None:
                    probs = run_inference(model, image)
                else:
                    probs = demo_prediction(uploaded.getvalue())

                order = np.argsort(probs)[::-1]
                top_idx = order[0]
                top_name = CLASS_NAMES[top_idx]
                crop, condition, is_healthy = parse_class_name(top_name)
                confidence = probs[top_idx] * 100
                hist_acc = PER_CLASS_ACCURACY.get(top_name)

                status_class = "status-healthy" if is_healthy else "status-disease"
                status_text = "Healthy" if is_healthy else "Condition detected"
                stamp_color = "var(--healthy)" if is_healthy else "var(--disease)"
                hist_line = (
                    f"""<p style="font-size:0.82rem; color:var(--ink-soft); margin-top:14px;">
                    On the held-out test set, this class was identified correctly
                    <b>{hist_acc:.1f}%</b> of the time.</p>"""
                    if hist_acc is not None
                    else ""
                )

                html_block(
                    f"""
                    <div class="result-wrap">
                        <div class="id-stamp" style="color:{stamp_color};">
                            <span class="num">{confidence:.0f}%</span>
                            <span class="unit">confidence</span>
                        </div>
                        <div class="result-eyebrow">Predicted class</div>
                        <p class="result-crop">{crop}</p>
                        <p class="result-condition">{condition if condition else '—'}</p>
                        <span class="status-tag {status_class}">{status_text}</span>
                        <div class="conf-row">
                            <div class="conf-label"><span>Model confidence</span><span>{confidence:.1f}%</span></div>
                            <div class="conf-track"><div class="conf-fill" style="width:{confidence:.1f}%;"></div></div>
                        </div>
                        {hist_line}
                    </div>
                    """
                )

                st.markdown(
                    '<div class="card-label" style="margin-top:22px;">Other candidates</div>',
                    unsafe_allow_html=True,
                )
                for idx in order[1:4]:
                    alt_crop, alt_cond, _ = parse_class_name(CLASS_NAMES[idx])
                    alt_p = probs[idx] * 100
                    st.markdown(
                        f"""
                        <div class="conf-label"><span>{alt_crop} — {alt_cond}</span><span>{alt_p:.1f}%</span></div>
                        <div class="conf-track"><div class="conf-fill alt" style="width:{alt_p:.1f}%;"></div></div>
                        """,
                        unsafe_allow_html=True,
                    )

                st.markdown(
                    '<div class="card-label" style="margin-top:22px;">Care notes</div>',
                    unsafe_allow_html=True,
                )
                for note in care_notes(top_name):
                    st.markdown(f"- {note}")

                st.markdown(
                    """<div class="caveat">Automated read from a CNN trained on a
                    fixed dataset — treat as a starting point, not a substitute
                    for an agronomist or extension office, especially before
                    applying any treatment.</div>""",
                    unsafe_allow_html=True,
                )

# ---------------------------------------------------------------------------
# Species index tab
# ---------------------------------------------------------------------------
with tab_index:
    st.markdown(
        '<p style="color:var(--ink-soft); font-size:0.9rem; margin-bottom:0;">'
        "Every condition the model was trained to recognize, grouped by crop, "
        "with its accuracy on the held-out test set.</p>",
        unsafe_allow_html=True,
    )

    by_crop = {}
    for name in CLASS_NAMES:
        crop, condition, is_healthy = parse_class_name(name)
        by_crop.setdefault(crop, []).append((condition or "healthy", PER_CLASS_ACCURACY[name]))

    cols = st.columns(2, gap="large")
    for i, crop in enumerate(sorted(by_crop.keys())):
        with cols[i % 2]:
            st.markdown(f'<div class="crop-heading">{crop}</div>', unsafe_allow_html=True)
            for condition, acc in by_crop[crop]:
                st.markdown(
                    f"""<div class="species-row">
                    <span class="cond">{condition}</span>
                    <span class="acc">{acc:.1f}%</span>
                    </div>""",
                    unsafe_allow_html=True,
                )
