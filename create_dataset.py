import os
from PIL import Image, ImageFilter, ImageOps

RAW_DIR = "dataset/raw"
OUTPUT_DIR = "dataset"

IMAGE_SIZE = 256

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


def create_sketch(image):
    """Convert a house photograph into a sketch-like input."""
    image = image.convert("RGB")
    image = ImageOps.grayscale(image)

    # Improve contrast
    image = ImageOps.autocontrast(image)

    # Detect edges
    edges = image.filter(ImageFilter.FIND_EDGES)

    # Invert so edges become dark on a light background
    edges = ImageOps.invert(edges)

    # Increase contrast again
    edges = ImageOps.autocontrast(edges)

    return edges.convert("RGB")


files = [
    f for f in os.listdir(RAW_DIR)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
]

print(f"Found {len(files)} house images.")

if len(files) == 0:
    print("\nNo images found.")
    print(f"Put house photographs inside: {RAW_DIR}")
    exit()


created = 0

for filename in files:

    path = os.path.join(RAW_DIR, filename)

    try:
        image = Image.open(path).convert("RGB")

        # Resize original house photograph
        target = image.resize(
            (IMAGE_SIZE, IMAGE_SIZE),
            Image.Resampling.LANCZOS
        )

        # Create sketch
        sketch = create_sketch(target)

        # Create side-by-side Pix2Pix image
        paired = Image.new(
            "RGB",
            (IMAGE_SIZE * 2, IMAGE_SIZE)
        )

        paired.paste(sketch, (0, 0))
        paired.paste(target, (IMAGE_SIZE, 0))

        output_name = f"paired_{created:04d}.jpg"
        output_path = os.path.join(
            OUTPUT_DIR,
            output_name
        )

        paired.save(output_path, quality=95)

        created += 1

        print(f"Created: {output_name}")

    except Exception as e:
        print(f"Skipping {filename}: {e}")


print("\n================================")
print("Dataset creation completed!")
print(f"Paired images created: {created}")
print("================================")