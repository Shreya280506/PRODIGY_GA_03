import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class Pix2PixDataset(Dataset):
    def __init__(self, root_dir, image_size=256):
        self.root_dir = root_dir

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.5, 0.5, 0.5),
                (0.5, 0.5, 0.5)
            )
        ])

        self.images = [
            file for file in os.listdir(root_dir)
            if file.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_path = os.path.join(
            self.root_dir,
            self.images[index]
        )

        image = Image.open(image_path).convert("RGB")

        width, height = image.size

        # Facades dataset:
        # LEFT  = real building photograph
        # RIGHT = semantic facade
        #
        # We want:
        # semantic facade -> realistic building
        #
        # Therefore:
        # input  = RIGHT half
        # target = LEFT half

        target_image = image.crop(
            (0, 0, width // 2, height)
        )

        input_image = image.crop(
            (width // 2, 0, width, height)
        )

        input_image = self.transform(input_image)
        target_image = self.transform(target_image)

        return input_image, target_image