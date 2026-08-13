import os
import io

import streamlit as st
import torch
from PIL import Image
import torchvision.transforms as transforms

from model import Generator


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="FacadeVision AI",
    page_icon="🏢",
    layout="wide"
)


# ============================================================
# SETTINGS
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMAGE_SIZE = 256

CHECKPOINT_PATH = (
    "checkpoints/generator.pth"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .stApp {
        background-color: #f4f1eb;
    }

    .main-title {
        font-size: 52px;
        font-weight: 700;
        letter-spacing: -2px;
        color: #111111;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 17px;
        color: #666666;
        margin-bottom: 45px;
    }

    .section-label {
        font-size: 13px;
        letter-spacing: 1px;
        color: #777777;
        text-transform: uppercase;
        margin-bottom: 8px;
    }

    .section-title {
        font-size: 30px;
        font-weight: 650;
        color: #111111;
        margin-bottom: 8px;
    }

    .description {
        font-size: 15px;
        color: #777777;
        margin-bottom: 20px;
    }

    .result-title {
        font-size: 30px;
        font-weight: 650;
        color: #111111;
        margin-bottom: 8px;
    }

    .status-box {
        padding: 14px 18px;
        border: 1px solid #d7d2c9;
        background-color: #ebe7df;
        border-radius: 6px;
        color: #555555;
        margin-bottom: 20px;
    }

    div.stButton > button {
        width: 100%;
        border-radius: 5px;
        height: 48px;
        font-size: 14px;
        font-weight: 600;
        background-color: #111111;
        color: white;
        border: none;
    }

    div.stButton > button:hover {
        background-color: #333333;
        color: white;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_generator():

    generator = Generator().to(DEVICE)

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE
    )

    generator.load_state_dict(checkpoint)

    generator.eval()

    return generator


# ============================================================
# IMAGE TRANSFORM
# ============================================================

transform = transforms.Compose([
    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.5, 0.5, 0.5),
        (0.5, 0.5, 0.5)
    )
])


# ============================================================
# GENERATION FUNCTION
# ============================================================

def generate_building(
    generator,
    image
):

    image = image.convert("RGB")

    input_tensor = transform(image)

    input_tensor = (
        input_tensor
        .unsqueeze(0)
        .to(DEVICE)
    )

    with torch.no_grad():

        generated = generator(
            input_tensor
        )

    generated = (
        generated + 1
    ) / 2

    generated = generated.clamp(
        0,
        1
    )

    generated = (
        generated
        .squeeze(0)
        .cpu()
    )

    output = transforms.ToPILImage()(
        generated
    )

    return output


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">FacadeVision AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle">
        Transform a semantic building layout into a realistic
        architectural façade using Pix2Pix.
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# MAIN COLUMNS
# ============================================================

left_column, right_column = st.columns(
    2,
    gap="large"
)


# ============================================================
# LEFT — INPUT
# ============================================================

with left_column:

    st.markdown(
        '<div class="section-label">01 / INPUT</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-title">Semantic Facade</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="description">
            Upload a semantic / label representation of a building.
        </div>
        """,
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader(
        "Upload your semantic facade",
        type=["jpg", "jpeg", "png"],
        label_visibility="visible"
    )

    input_image = None

    if uploaded_file is not None:

        input_image = Image.open(
            io.BytesIO(
                uploaded_file.getvalue()
            )
        ).convert("RGB")

        st.markdown(
            '<div class="section-label">INPUT PREVIEW · 256 × 256</div>',
            unsafe_allow_html=True
        )

        st.image(
            input_image,
            width=256
        )


# ============================================================
# RIGHT — OUTPUT
# ============================================================

with right_column:

    st.markdown(
        '<div class="section-label">02 / OUTPUT</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="result-title">Realistic Building</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="description">
            Generated architectural visualization.
        </div>
        """,
        unsafe_allow_html=True
    )

    if (
        "generated_image"
        in st.session_state
    ):

        st.markdown(
            '<div class="section-label">OUTPUT PREVIEW · 256 × 256</div>',
            unsafe_allow_html=True
        )

        st.image(
            st.session_state.generated_image,
            width=256
        )

    else:

        st.markdown(
            """
            <div class="status-box">
                Upload a semantic facade and click
                <b>GENERATE REALISTIC BUILDING</b>.
            </div>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# GENERATE BUTTON
# ============================================================

st.write("")

if uploaded_file is not None:

    if st.button(
        "GENERATE REALISTIC BUILDING"
    ):

        try:

            with st.spinner(
                "Generating realistic building..."
            ):

                generator = load_generator()

                result = generate_building(
                    generator,
                    input_image
                )

                st.session_state.generated_image = result

            st.rerun()

        except Exception as e:

            st.error(
                f"Generation failed: {e}"
            )


# ============================================================
# FOOTER
# ============================================================

st.write("")

st.markdown(
    """
    <div style="
        margin-top:50px;
        padding-top:20px;
        border-top:1px solid #d7d2c9;
        color:#777777;
        font-size:13px;
    ">
        Pix2Pix Conditional GAN · U-Net Generator · PatchGAN Discriminator
    </div>
    """,
    unsafe_allow_html=True
)
