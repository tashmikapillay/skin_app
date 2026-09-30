import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms, models
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import matplotlib.pyplot as plt

# Page config 
st.set_page_config(
    page_title="Skin Disease Detection",
    page_icon="🔬",
    layout="wide",
)

# Constants 
IMG_SIZE    = 448
MEAN        = [0.485, 0.456, 0.406]
STD         = [0.229, 0.224, 0.225]
CLASS_NAMES = ['mel', 'nv', 'bkl', 'bcc', 'akiec', 'vasc', 'df']
CLASS_FULL  = {
    'mel':   'Melanoma',
    'nv':    'Melanocytic Nevus (Mole)',
    'bkl':   'Benign Keratosis',
    'bcc':   'Basal Cell Carcinoma',
    'akiec': 'Actinic Keratosis / Intraepithelial Carcinoma',
    'vasc':  'Vascular Lesion',
    'df':    'Dermatofibroma',
}
CLASS_RISK = {
    'mel':   '🔴 High Risk — Malignant',
    'nv':    '🟢 Low Risk — Benign',
    'bkl':   '🟢 Low Risk — Benign',
    'bcc':   '🔴 High Risk — Malignant',
    'akiec': '🟡 Moderate Risk — Pre-malignant',
    'vasc':  '🟡 Moderate Risk',
    'df':    '🟢 Low Risk — Benign',
}
ABCDE_NOTES = {
    'mel':   'Activations on dark pigmented body and irregular border — aligns with A (asymmetry) and C (colour variation).',
    'nv':    'Diffuse activations across the lesion — consistent with symmetric, uniformly pigmented mole.',
    'bkl':   'Activations on rough surface texture — aligns with D (differential structures, milia-like cysts).',
    'bcc':   'Focal activations on internal structures — aligns with B (irregular border) and D (arborising vessels).',
    'akiec': 'Activations on scaly surface patches — aligns with D (keratotic scaling) and B (poorly defined border).',
    'vasc':  'Concentrated activation on vascular structure — aligns with D (red lacunae and vascular structures).',
    'df':    'Activations on lesion border and peripheral region — aligns with B (border) and A (asymmetry).',
}

# CLIP uses two prompts: dermoscopic lesion vs not a lesion
CLIP_POSITIVE = "a close-up dermoscopic image of a skin lesion or mole"
CLIP_NEGATIVE = "a regular photo of normal skin, a mosquito bite, an insect sting, a bruise, or a non-lesion"

MODEL_PATH = 'hybrid_best.pth'


#  Model definition 
class LightweightHybridModel(nn.Module):
    def __init__(self, num_classes=7, embed_dim=384, num_heads=4,
                 num_transformer_layers=4, dropout=0.1, img_size=448):
        super().__init__()
        mobilenet = models.mobilenet_v3_small(weights=None)
        self.cnn_backbone = mobilenet.features
        with torch.no_grad():
            _dummy = torch.zeros(1, 3, img_size, img_size)
            _feat  = mobilenet.features(_dummy)
            num_patches = _feat.shape[2] * _feat.shape[3]
        self.patch_proj    = nn.Linear(576, embed_dim)
        self.cls_token     = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed     = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.dropout_embed = nn.Dropout(dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout, activation='gelu',
            batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(embed_dim, num_classes))

    def forward(self, x):
        feat = self.cnn_backbone(x)
        B, C, H, W = feat.shape
        tokens = self.patch_proj(feat.flatten(2).transpose(1, 2))
        cls    = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        tokens = self.dropout_embed(tokens + self.pos_embed)
        tokens = self.transformer(tokens)
        tokens = self.norm(tokens)
        return self.head(tokens[:, 0])


# Load hybrid model 
@st.cache_resource
def load_model():
    model = LightweightHybridModel(num_classes=7, embed_dim=384, img_size=IMG_SIZE, dropout=0.3)
    state = torch.load(MODEL_PATH, map_location='cpu')
    model.load_state_dict(state)
    model.eval()
    return model


# Load CLIP model 
@st.cache_resource
def load_clip():
    from transformers import CLIPProcessor, CLIPModel
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_proc  = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip_model.eval()
    return clip_model, clip_proc


# Transforms 
preprocess = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])
unnorm = transforms.Normalize(
    mean=[-m / s for m, s in zip(MEAN, STD)],
    std=[1 / s for s in STD],
)


# Hair removal 
def remove_hair(img_rgb: np.ndarray) -> np.ndarray:
    img_bgr  = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    gray     = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    kernel   = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    _, mask  = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    cleaned  = cv2.inpaint(img_bgr, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    return cv2.cvtColor(cleaned, cv2.COLOR_BGR2RGB)


# CLIP: check if image is a skin lesion 
def is_skin_lesion(clip_model, clip_proc, pil_img: Image.Image):
    """Returns (is_lesion: bool, lesion_confidence: float)"""
    inputs = clip_proc(
        text=[CLIP_POSITIVE, CLIP_NEGATIVE],
        images=pil_img,
        return_tensors="pt",
        padding=True,
    )
    with torch.no_grad():
        logits = clip_model(**inputs).logits_per_image[0]
        probs  = torch.softmax(logits, dim=0).numpy()
    return bool(probs[0] > probs[1]), float(probs[0])


# Hybrid model inference + Grad-CAM 
def predict(model, img_rgb: np.ndarray):
    cleaned   = remove_hair(img_rgb)
    pil_img   = Image.fromarray(cleaned)
    tensor    = preprocess(pil_img).unsqueeze(0)

    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1)[0].numpy()

    pred_idx  = int(np.argmax(probs))
    pred_name = CLASS_NAMES[pred_idx]

    cam = GradCAM(model=model, target_layers=[model.cnn_backbone[-1]])
    grayscale_cam = cam(input_tensor=tensor,
                        targets=[ClassifierOutputTarget(pred_idx)])[0]
    rgb_norm  = np.clip(unnorm(preprocess(pil_img)).permute(1, 2, 0).numpy(), 0, 1)
    overlay   = show_cam_on_image(rgb_norm, grayscale_cam, use_rgb=True)

    return pred_name, probs, cleaned, overlay

# UI
st.title("Automated Skin Disease Detection")
st.markdown("**Author:** Tashmika Pillay | **Supervisor:** Prof Serestina Viriri | University of KwaZulu-Natal")

st.warning(
    "⚠️ **Research Prototype Disclaimer:** This system is not validated for clinical use. "
    "All clinical decisions must be made by a qualified medical professional. "
    "This tool is intended for research and educational purposes only."
)

st.markdown("---")

# Load models
try:
    model = load_model()
except FileNotFoundError:
    st.error(f"Model file `{MODEL_PATH}` not found. Place `hybrid_best.pth` in the same folder as `app.py`.")
    st.stop()

clip_available = True
try:
    clip_model, clip_proc = load_clip()
except Exception:
    clip_available = False

uploaded = st.file_uploader("Upload a dermoscopic image", type=["jpg", "jpeg", "png"])

if uploaded:
    img_pil = Image.open(uploaded).convert("RGB")
    img_rgb = np.array(img_pil)

    # Step 1: CLIP OOD check 
    if clip_available:
        with st.spinner("Checking image validity with CLIP..."):
            lesion, lesion_conf = is_skin_lesion(clip_model, clip_proc, img_pil)

        if not lesion:
            col1, _ = st.columns([1, 2])
            with col1:
                st.subheader("Uploaded Image")
                st.image(img_rgb, use_container_width=True)
            st.error(
                f"**Image not recognised as a skin lesion** (CLIP confidence: {lesion_conf*100:.1f}% lesion). "
                "This image does not appear to be a dermoscopic skin lesion image. "
                "It may be a mosquito bite, bruise, rash, or regular photo. "
                "Please upload a proper dermoscopic image for accurate classification."
            )
            st.info(
                "**Why does this happen?** This hybrid model was trained only on dermoscopic images "
                "from the HAM10000 dataset. CLIP (zero-shot vision-language model) is used as a "
                "pre-screening step to detect and reject images outside this distribution."
            )
            st.stop()

    # Step 2: Hybrid model inference
    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("Original Image")
        st.image(img_rgb, use_container_width=True)

    with st.spinner("Analysing image..."):
        pred_name, probs, cleaned, overlay = predict(model, img_rgb)

    with col2:
        st.subheader("Hair Removed")
        st.image(cleaned, use_container_width=True)

    with col3:
        st.subheader("Grad-CAM Explanation")
        st.image(overlay, use_container_width=True)

    if clip_available:
        st.caption(f"✅ CLIP pre-screen passed — image recognised as a dermoscopic skin lesion ({lesion_conf*100:.1f}% confidence).")

    st.markdown("---")

    pred_full = CLASS_FULL[pred_name]
    risk      = CLASS_RISK[pred_name]
    conf      = float(probs[CLASS_NAMES.index(pred_name)]) * 100

    st.subheader("Prediction")
    res_col1, res_col2 = st.columns(2)

    with res_col1:
        st.metric("Predicted Diagnosis", pred_full)
        st.metric("Confidence", f"{conf:.1f}%")
        st.markdown(f"**Risk Level:** {risk}")

    with res_col2:
        st.subheader("Confidence Scores")
        fig, ax = plt.subplots(figsize=(5, 3))
        colors  = ['#d62728' if n == pred_name else '#1f77b4' for n in CLASS_NAMES]
        ax.barh([CLASS_FULL[n] for n in CLASS_NAMES],
                [probs[i] * 100 for i in range(len(CLASS_NAMES))],
                color=colors)
        ax.set_xlabel("Confidence (%)")
        ax.set_xlim(0, 100)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown("---")
    st.subheader("Clinical Explanation (ABCDE Alignment)")
    st.info(f"**{pred_full}:** {ABCDE_NOTES[pred_name]}")
    st.markdown("""
    The Grad-CAM heatmap highlights the image regions the model focused on when making this prediction.
    Warmer colours (red/yellow) indicate higher attention. These regions are assessed against the
    **ABCDE rule** used by dermatologists: Asymmetry, Border, Colour, Diameter, Evolution.
    """)

    st.markdown("---")
    st.caption("Lightweight Hybrid CNN-Transformer Model | HAM10000 Dataset | UKZN Honours Project 2026")
