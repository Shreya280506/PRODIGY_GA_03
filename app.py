import os
import torch
import streamlit as st
from PIL import Image
import torchvision.transforms as transforms

from model import Generator
from config import DEVICE, IMAGE_SIZE, GENERATOR_CHECKPOINT


# --------------------------------
# Page Configuration
# --------------------------------

st.set_page_config(
    page_title="Sketch to Realistic House",
    page_icon="🏠",
    layout="wide"
)


# --------------------------------
# Load Model
# --------------------------------

@st.cache_resource
def load_model():

    generator = Generator().to(DEVICE)

    generator.load_state_dict(
        torch.load(
            GENERATOR_CHECKPOINT,
            map_location=DEVICE
        )
    )

    generator.eval()

    return generator


# --------------------------------
# Image Translation
# --------------------------------

def translate_image(generator, input_image):

    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.5, 0.5, 0.5),
            (0.5, 0.5, 0.5)
        )
    ])

    image_tensor = transform(input_image)
    image_tensor = image_tensor.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        generated = generator(image_tensor)

    # Convert from [-1, 1] to [0, 1]
    generated = (generated + 1) / 2

    generated = generated.squeeze(0).cpu()

    generated_image = transforms.ToPILImage()(generated)

    return generated_image


# --------------------------------
# Header
# --------------------------------

st.title("🏠 Sketch → Realistic House")

st.markdown(
    """
    ### Generate a realistic house from a sketch

    Upload a **house sketch or edge-style drawing**, and the trained
    Pix2Pix model will generate a realistic house image.
    """
)

st.info(
    f"Model: Pix2Pix cGAN | "
    f"Training images: 500 | "
    f"Image size: {IMAGE_SIZE}×{IMAGE_SIZE} | "
    f"Device: {DEVICE.upper()}"
)


# --------------------------------
# Load Generator
# --------------------------------

generator = load_model()


# --------------------------------
# Upload Image
# --------------------------------

uploaded_file = st.file_uploader(
    "📤 Upload a house sketch",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:

    input_image = Image.open(
        uploaded_file
    ).convert("RGB")

    st.subheader("Input Sketch")

    st.image(
        input_image,
        width="stretch"
    )

    # --------------------------------
    # Generate
    # --------------------------------

    if st.button(
        "✨ Generate Realistic House",
        type="primary"
    ):

        with st.spinner(
            "Generating realistic house..."
        ):

            generated_image = translate_image(
                generator,
                input_image
            )

        st.success(
            "Realistic house generated successfully!"
        )

        # --------------------------------
        # Results
        # --------------------------------

        st.subheader("Result")

        col1, col2 = st.columns(2)

        with col1:

            st.markdown("### ✏️ Input Sketch")

            st.image(
                input_image,
                width="stretch"
            )

        with col2:

            st.markdown("### 🏠 Generated House")

            st.image(
                generated_image,
                width="stretch"
            )

        # --------------------------------
        # Save Output
        # --------------------------------

        os.makedirs(
            "outputs",
            exist_ok=True
        )

        output_path = (
            "outputs/streamlit_output.png"
        )

        generated_image.save(
            output_path
        )

        # --------------------------------
        # Download
        # --------------------------------

        with open(
            output_path,
            "rb"
        ) as file:

            st.download_button(
                label="⬇️ Download Generated House",
                data=file,
                file_name="realistic_house.png",
                mime="image/png"
            )


# --------------------------------
# About
# --------------------------------

st.divider()

st.subheader("ℹ️ About the Project")

st.write(
    """
    This project uses **Pix2Pix**, a conditional Generative
    Adversarial Network (cGAN), for paired image-to-image translation.

    The model was trained to learn the mapping:

    **House Sketch → Realistic House**

    The Generator uses a U-Net architecture to transform the
    input sketch, while the PatchGAN Discriminator learns to
    distinguish realistic target images from generated images.
    """
)