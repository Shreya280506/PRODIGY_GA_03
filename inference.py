import os
import torch
from PIL import Image
import torchvision.transforms as transforms

from model import Generator
from config import DEVICE, IMAGE_SIZE, GENERATOR_CHECKPOINT


# Create output directory
os.makedirs("outputs", exist_ok=True)


# Load generator
generator = Generator().to(DEVICE)

generator.load_state_dict(
    torch.load(
        GENERATOR_CHECKPOINT,
        map_location=DEVICE
    )
)

generator.eval()


# Select an input image
input_path = "dataset/100.jpg"

image = Image.open(input_path).convert("RGB")

# The dataset image contains:
# left  = input
# right = target

width, height = image.size

input_image = image.crop(
    (0, 0, width // 2, height)
)


# Transform input
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.5, 0.5, 0.5),
        (0.5, 0.5, 0.5)
    )
])

input_tensor = transform(input_image)
input_tensor = input_tensor.unsqueeze(0).to(DEVICE)


# Generate image
with torch.no_grad():
    generated = generator(input_tensor)


# Convert from [-1, 1] to [0, 1]
generated = (generated + 1) / 2

generated = generated.squeeze(0).cpu()

generated_image = transforms.ToPILImage()(generated)

# Save output
output_path = "outputs/generated.png"

generated_image.save(output_path)

print(f"Generated image saved to: {output_path}")