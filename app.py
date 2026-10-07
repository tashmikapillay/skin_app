import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torchvision import models, transforms

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
# Kind of lesion each class represents (text, colour used to display it)
CLASS_TYPE = {
    'mel':   ('Malignant', 'red'),
    'bcc':   ('Malignant', 'red'),
    'akiec': ('Precancerous or early (in situ) cancer', 'orange'),
    'nv':    ('Benign', 'green'),
    'bkl':   ('Benign', 'green'),
    'vasc':  ('Benign', 'green'),
    'df':    ('Benign', 'green'),
}
# Share of test images of each class that the model classified correctly (HAM10000 test split, 1,516 images)
TEST_SENSITIVITY = {'mel': 0.57, 'nv': 0.92, 'bkl': 0.55, 'bcc': 0.69, 'akiec': 0.75, 'vasc': 0.94, 'df': 0.70}

# CLIP compares two prompts: a dermoscopic lesion against an ordinary photo
CLIP_POSITIVE = "a close-up dermoscopic image of a skin lesion or mole"
CLIP_NEGATIVE = "a regular photo of normal skin, a mosquito bite, an insect sting, a bruise, or a non-lesion"

APP_DIR      = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE   = 'hybrid_v2_best.pth'
MODEL_PATH   = os.path.join(APP_DIR, MODEL_FILE)
# HAM10000 images are 600 x 450. The hair removal settings are tuned for that scale,
# so every upload is brought to it first, exactly as the training images were.
SOURCE_SIZE  = (600, 450)   # (width, height)


# Model definition (must match the training notebook)
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
    from transformers import CLIPModel, CLIPProcessor
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_proc  = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip_model.eval()
    return clip_model, clip_proc


# Transforms (the same as the validation transform used in training)
preprocess = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])
unnorm = transforms.Normalize(
    mean=[-m / s for m, s in zip(MEAN, STD)],
    std=[1 / s for s in STD],
)


# Hair removal (the conservative version used to train the model)
def hair_mask(img_bgr: np.ndarray, kernel_size: int = 11, threshold: int = 22, min_length: int = 35) -> np.ndarray:
    """Mask of hair pixels: dark, thin structures that are long, and thin or sparse inside their bounding box."""
    gray     = cv2.GaussianBlur(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY), (3, 3), 0)
    kernel   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    cand     = (blackhat > threshold).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(cand, connectivity=8)
    w, h, area = stats[:, cv2.CC_STAT_WIDTH], stats[:, cv2.CC_STAT_HEIGHT], stats[:, cv2.CC_STAT_AREA]
    length = np.maximum(w, h)
    fill   = area / np.maximum(w * h, 1)
    aspect = length / np.maximum(np.minimum(w, h), 1)
    keep   = (length >= min_length) & ((fill < 0.35) | (aspect >= 4))
    keep[0] = False                                   # label 0 is the background
    mask = (keep[labels] * 255).astype(np.uint8)
    return cv2.dilate(mask, np.ones((3, 3), np.uint8))


def remove_hair(img_rgb: np.ndarray):
    """Returns the cleaned image (at the HAM10000 scale) and the share of pixels that were repainted."""
    if (img_rgb.shape[1], img_rgb.shape[0]) != SOURCE_SIZE:
        shrinking = img_rgb.shape[1] > SOURCE_SIZE[0]
        img_rgb = cv2.resize(img_rgb, SOURCE_SIZE, interpolation=cv2.INTER_AREA if shrinking else cv2.INTER_LINEAR)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    mask    = hair_mask(img_bgr)
    if not mask.any():
        return img_rgb, 0.0
    cleaned = cv2.inpaint(img_bgr, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    return cv2.cvtColor(cleaned, cv2.COLOR_BGR2RGB), float((mask > 0).mean())


# CLIP: basic check that the upload looks like a skin lesion image
def is_skin_lesion(clip_model, clip_proc, pil_img: Image.Image):
    """Returns (is_lesion: bool, lesion_score: float)"""
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
    cleaned, hair_share = remove_hair(img_rgb)
    pil_img = Image.fromarray(cleaned)
    tensor  = preprocess(pil_img).unsqueeze(0)

    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1)[0].numpy()

    pred_idx  = int(np.argmax(probs))
    pred_name = CLASS_NAMES[pred_idx]

    cam = GradCAM(model=model, target_layers=[model.cnn_backbone[-1]])
    try:
        grayscale_cam = cam(input_tensor=tensor,
                            targets=[ClassifierOutputTarget(pred_idx)])[0]
    finally:
        cam.activations_and_grads.release()   # the model is cached, so the hooks must not pile up between uploads
    rgb_norm = np.clip(unnorm(preprocess(pil_img)).permute(1, 2, 0).numpy(), 0, 1)
    overlay  = show_cam_on_image(rgb_norm, grayscale_cam, use_rgb=True)
    overlay  = cv2.resize(overlay, (cleaned.shape[1], cleaned.shape[0]))   # show it at the same shape as the image beside it

    return pred_name, probs, cleaned, hair_share, overlay


# UI
st.title("Automated Skin Disease Detection")
st.markdown("**Author:** Tashmika Pillay | **Supervisor:** Prof Serestina Viriri | University of KwaZulu-Natal")

st.warning(
    "**Research prototype.** This system is not validated for clinical use. "
    "All clinical decisions must be made by a qualified medical professional. "
    "This tool is intended for research and educational purposes only."
)

with st.expander("About this model and how reliable it is"):
    st.markdown(
        "The classifier is a hybrid of a MobileNetV3-Small network and a four layer Transformer encoder "
        "(8.3 million parameters), trained on the HAM10000 dermoscopic dataset at 448 x 448 pixels.\n\n"
        "**Results on 1,516 held out test images** (lesions never seen in training):\n\n"
        "| Accuracy | Balanced accuracy | Macro F1 | Macro AUROC |\n"
        "|---|---|---|---|\n"
        "| 0.821 | 0.729 | 0.699 | 0.946 |\n\n"
        "Always answering *nevus* would score an accuracy of 0.672 on the same images.\n\n"
        "**Share of each class the model identified correctly:** "
        + ", ".join(f"{CLASS_FULL[c]} {TEST_SENSITIVITY[c]:.0%}" for c in CLASS_NAMES) + ".\n\n"
        "**Important limits**\n"
        "- The model missed 43% of the melanomas in the test set, and most of those were labelled as nevus. "
        "A benign result does not rule out melanoma.\n"
        "- HAM10000 consists mainly of images of lighter skin. Performance on other skin tones is unknown "
        "and likely to be lower.\n"
        "- The model expects dermoscopic images. It was not trained on ordinary phone photographs.\n"
        "- The input check that runs before classification accepted 98% of real lesion images in testing, "
        "but rejected only about half of unrelated photographs."
    )

st.markdown("---")

# Load models
try:
    model = load_model()
except FileNotFoundError:
    st.error(f"Model file `{MODEL_FILE}` not found. Place it in the same folder as `app.py`.")
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

    # Step 1: input check with CLIP
    if clip_available:
        with st.spinner("Checking the image..."):
            lesion, lesion_score = is_skin_lesion(clip_model, clip_proc, img_pil)

        if not lesion:
            st.error(
                "**This image did not pass the input check.** It does not look like a dermoscopic image "
                "of a skin lesion. The classifier was trained only on dermoscopic images, so its answer "
                "for any other kind of picture would not be meaningful."
            )
            st.caption("The check is not perfect. In testing it wrongly turned away about 2% of real lesion images.")
            if not st.checkbox("This is a dermoscopic image. Classify it anyway."):
                col1, _ = st.columns([1, 2])
                with col1:
                    st.subheader("Uploaded Image")
                    st.image(img_rgb, width="stretch")
                st.stop()

    # Step 2: hybrid model inference
    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("Original Image")
        st.image(img_rgb, width="stretch")

    with st.spinner("Analysing image..."):
        pred_name, probs, cleaned, hair_share, overlay = predict(model, img_rgb)

    with col2:
        st.subheader("After Hair Removal")
        st.image(cleaned, width="stretch")
        if hair_share == 0:
            st.caption("No hair was detected, so the image was left unchanged.")
        else:
            st.caption(f"Hair was detected and {hair_share:.1%} of the image was repainted.")

    with col3:
        st.subheader("Grad-CAM Explanation")
        st.image(overlay, width="stretch")
        st.caption("Warmer colours mark the regions that most raised the score of the predicted class.")

    if clip_available and lesion:
        st.caption(
            "Input check passed. This check rejected only about half of unrelated photographs in testing, "
            "so a pass does not confirm that the image is a dermoscopic lesion."
        )
    elif not clip_available:
        st.caption("The input check is unavailable, so the image was classified without it.")

    st.markdown("---")

    pred_full = CLASS_FULL[pred_name]
    type_text, type_colour = CLASS_TYPE[pred_name]
    conf      = float(probs[CLASS_NAMES.index(pred_name)]) * 100

    st.subheader("Prediction")
    res_col1, res_col2 = st.columns(2)

    with res_col1:
        st.metric("Predicted class", pred_full)
        st.metric("Model probability", f"{conf:.1f}%")
        st.markdown(f"**Type of lesion:** :{type_colour}[{type_text}]")
        st.caption(
            f"On the test set the model correctly identified {TEST_SENSITIVITY[pred_name]:.0%} "
            f"of the images that truly were {pred_full.lower()}."
        )
        if CLASS_TYPE[pred_name][0] == 'Benign':
            st.info(
                "A benign prediction does not rule out melanoma. "
                "In testing, 43 of 145 melanomas were predicted as nevus."
            )

    with res_col2:
        st.subheader("Probability of Each Class")
        fig, ax = plt.subplots(figsize=(5, 3))
        colors  = ['#d62728' if n == pred_name else '#1f77b4' for n in CLASS_NAMES]
        ax.barh([CLASS_FULL[n] for n in CLASS_NAMES],
                [probs[i] * 100 for i in range(len(CLASS_NAMES))],
                color=colors)
        ax.set_xlabel("Probability (%)")
        ax.set_xlim(0, 100)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown("---")
    st.subheader("How to Read the Explanation")
    st.markdown(
        "The Grad-CAM heatmap shows which parts of the image most increased the model's score for the "
        "predicted class. It describes what this model responded to. It is not a clinical finding.\n\n"
        "- In testing, the highlighted regions were enough to restore the model's prediction more "
        "efficiently than a plain blob over the image centre, so the maps carry real information "
        "about the model.\n"
        "- Removing the highlighted regions did not lower the prediction faster than removing the "
        "centre, so the maps do not prove that those regions are the ones the model needs.\n"
        "- The explanations have not been reviewed by dermatologists, and they do not show that the model "
        "uses the features a clinician would use."
    )

    st.markdown("---")
    st.caption("Lightweight Hybrid CNN-Transformer Model | HAM10000 Dataset | UKZN Honours Project 2026")
