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
            file
            for file in os.listdir(root_dir)
            if file.lower().endswith(
                (".jpg", ".jpeg", ".png")
            )
        ]

        self.images.sort()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):

        image_path = os.path.join(
            self.root_dir,
            self.images[index]
        )

        image = Image.open(image_path).convert("RGB")

        width, height = image.size

        # =====================================================
        # FACADES DATASET
        # =====================================================
        #
        # LEFT  = REAL BUILDING PHOTOGRAPH
        # RIGHT = SEMANTIC / LABEL REPRESENTATION
        #
        # We want:
        #
        # INPUT  = semantic representation
        # TARGET = realistic building
        #
        # Therefore we REVERSE the usual left/right order.
        # =====================================================

        real_building = image.crop(
            (0, 0, width // 2, height)
        )

        semantic_label = image.crop(
            (width // 2, 0, width, height)
        )

        # INPUT = semantic representation
        input_image = self.transform(semantic_label)

        # TARGET = real building photograph
        target_image = self.transform(real_building)

        return input_image, target_image