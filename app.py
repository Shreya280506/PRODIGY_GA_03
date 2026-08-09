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
    page_title="Pix2Pix Image Translator",
    page_icon="🎨",
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

    generated = (generated + 1) / 2
    generated = generated.squeeze(0).cpu()

    return transforms.ToPILImage()(generated)


# --------------------------------
# Header
# --------------------------------

st.title("🎨 Pix2Pix Image Translator")

st.markdown(
    """
    ### Image-to-Image Translation using Conditional GAN

    Upload a **facade sketch/label image** and Pix2Pix will
    generate a corresponding realistic facade image.
    """
)

st.info(
    "Model: Pix2Pix cGAN | Dataset: CMP Facades | "
    f"Device: {DEVICE.upper()}"
)


# --------------------------------
# Load Generator
# --------------------------------

generator = load_model()


# --------------------------------
# Upload
# --------------------------------

uploaded_file = st.file_uploader(
    "📤 Upload an input image",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:

    input_image = Image.open(
        uploaded_file
    ).convert("RGB")

    st.subheader("Input Image")

    st.image(
        input_image,
        width="stretch"
    )

    if st.button(
        "✨ Generate Translation",
        type="primary"
    ):

        with st.spinner(
            "Generating translated image..."
        ):

            generated_image = translate_image(
                generator,
                input_image
            )

        st.success(
            "Image generated successfully!"
        )

        # --------------------------------
        # Results
        # --------------------------------

        st.subheader("Translation Result")

        col1, col2 = st.columns(2)

        with col1:

            st.markdown("### 📥 Input")

            st.image(
                input_image,
                width="stretch"
            )

        with col2:

            st.markdown("### 🎨 Generated")

            st.image(
                generated_image,
                width="stretch"
            )

        # --------------------------------
        # Save
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
                label="⬇️ Download Generated Image",
                data=file,
                file_name="pix2pix_generated.png",
                mime="image/png"
            )


# --------------------------------
# About
# --------------------------------

st.divider()

st.subheader("ℹ️ About Pix2Pix")

st.write(
    """
    Pix2Pix is a conditional Generative Adversarial Network
    designed for paired image-to-image translation.

    The Generator uses a U-Net architecture to transform
    the input image, while the PatchGAN Discriminator learns
    to distinguish real target images from generated images.
    """
)