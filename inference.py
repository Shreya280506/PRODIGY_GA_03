import os
import torch
from PIL import Image
import torchvision.transforms as transforms

from model import Generator


# ============================================================
# SETTINGS
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMAGE_SIZE = 256

CHECKPOINT = "checkpoints_facades/generator.pth"

INPUT_IMAGE = "facades_right.jpg"

OUTPUT_IMAGE = "outputs/facades_generated.png"


# ============================================================
# OUTPUT DIRECTORY
# ============================================================

os.makedirs("outputs", exist_ok=True)


# ============================================================
# LOAD GENERATOR
# ============================================================

print("Using device:", DEVICE)
print("Loading Facades generator...")

generator = Generator().to(DEVICE)

generator.load_state_dict(
    torch.load(
        CHECKPOINT,
        map_location=DEVICE
    )
)

generator.eval()

print("Generator loaded successfully.")


# ============================================================
# LOAD SEMANTIC INPUT
# ============================================================

image = Image.open(INPUT_IMAGE).convert("RGB")


# ============================================================
# TRANSFORM
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


input_tensor = transform(image)
input_tensor = input_tensor.unsqueeze(0).to(DEVICE)


# ============================================================
# GENERATE
# ============================================================

print("Generating realistic building...")

with torch.no_grad():

    generated = generator(input_tensor)


# ============================================================
# CONVERT [-1,1] → [0,1]
# ============================================================

generated = (generated + 1) / 2

generated = generated.clamp(0, 1)

generated = generated.squeeze(0).cpu()


# ============================================================
# SAVE
# ============================================================

generated_image = transforms.ToPILImage()(generated)

generated_image.save(
    OUTPUT_IMAGE
)


# ============================================================
# COMPLETE
# ============================================================

print()
print("==============================================")
print("FACADES INFERENCE COMPLETED")
print("==============================================")
print("Input :", INPUT_IMAGE)
print("Output:", OUTPUT_IMAGE)
print("==============================================")