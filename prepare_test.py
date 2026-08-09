from PIL import Image

image = Image.open("dataset/100.jpg")

width, height = image.size

input_image = image.crop(
    (0, 0, width // 2, height)
)

input_image.save("test_input.jpg")

print("Test input saved as test_input.jpg")