import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import (
    DEVICE,
    BATCH_SIZE,
    LEARNING_RATE,
    NUM_EPOCHS,
    BETA1,
    BETA2,
    LAMBDA_L1,
    IMAGE_SIZE
)

from dataset import Pix2PixDataset
from model import Generator, Discriminator


# ============================================================
# SETTINGS
# ============================================================

DATASET_DIR = "facades/train"
CHECKPOINT_DIR = "checkpoints_facades"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

GENERATOR_CHECKPOINT = os.path.join(
    CHECKPOINT_DIR,
    "generator.pth"
)

DISCRIMINATOR_CHECKPOINT = os.path.join(
    CHECKPOINT_DIR,
    "discriminator.pth"
)


# ============================================================
# DATASET
# ============================================================

dataset = Pix2PixDataset(
    root_dir=DATASET_DIR,
    image_size=IMAGE_SIZE
)

dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)


# ============================================================
# MODELS
# ============================================================

generator = Generator().to(DEVICE)
discriminator = Discriminator().to(DEVICE)


# ============================================================
# LOSS FUNCTIONS
# ============================================================

adversarial_loss = nn.BCEWithLogitsLoss()
l1_loss = nn.L1Loss()


# ============================================================
# OPTIMIZERS
# ============================================================

optimizer_G = torch.optim.Adam(
    generator.parameters(),
    lr=LEARNING_RATE,
    betas=(BETA1, BETA2)
)

optimizer_D = torch.optim.Adam(
    discriminator.parameters(),
    lr=LEARNING_RATE,
    betas=(BETA1, BETA2)
)


# ============================================================
# TRAINING INFORMATION
# ============================================================

print()
print("================================================")
print("PIX2PIX FACADES TRAINING")
print("================================================")
print("Task:")
print("Semantic / Label → Realistic Building")
print()
print(f"Device: {DEVICE}")
print(f"Training images: {len(dataset)}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Epochs: {NUM_EPOCHS}")
print(f"Image size: {IMAGE_SIZE}x{IMAGE_SIZE}")
print(f"L1 weight: {LAMBDA_L1}")
print()
print("Input  = Semantic / Label")
print("Target = Real Building")
print("================================================")
print()


# ============================================================
# TRAINING LOOP
# ============================================================

for epoch in range(NUM_EPOCHS):

    progress_bar = tqdm(
        dataloader,
        desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}"
    )

    total_generator_loss = 0.0
    total_discriminator_loss = 0.0

    for input_image, target_image in progress_bar:

        input_image = input_image.to(DEVICE)
        target_image = target_image.to(DEVICE)

        # ====================================================
        # TRAIN GENERATOR
        # ====================================================

        optimizer_G.zero_grad()

        fake_image = generator(input_image)

        fake_prediction = discriminator(
            input_image,
            fake_image
        )

        gan_loss = adversarial_loss(
            fake_prediction,
            torch.ones_like(fake_prediction)
        )

        reconstruction_loss = l1_loss(
            fake_image,
            target_image
        )

        generator_loss = (
            gan_loss +
            LAMBDA_L1 * reconstruction_loss
        )

        generator_loss.backward()

        optimizer_G.step()


        # ====================================================
        # TRAIN DISCRIMINATOR
        # ====================================================

        optimizer_D.zero_grad()

        # Real pair
        real_prediction = discriminator(
            input_image,
            target_image
        )

        real_loss = adversarial_loss(
            real_prediction,
            torch.ones_like(real_prediction)
        )

        # Fake pair
        fake_prediction = discriminator(
            input_image,
            fake_image.detach()
        )

        fake_loss = adversarial_loss(
            fake_prediction,
            torch.zeros_like(fake_prediction)
        )

        discriminator_loss = (
            real_loss +
            fake_loss
        ) * 0.5

        discriminator_loss.backward()

        optimizer_D.step()


        # ====================================================
        # TRACK LOSSES
        # ====================================================

        total_generator_loss += generator_loss.item()
        total_discriminator_loss += discriminator_loss.item()

        progress_bar.set_postfix(
            G_loss=f"{generator_loss.item():.4f}",
            D_loss=f"{discriminator_loss.item():.4f}"
        )


    # ========================================================
    # AVERAGE LOSSES
    # ========================================================

    average_generator_loss = (
        total_generator_loss / len(dataloader)
    )

    average_discriminator_loss = (
        total_discriminator_loss / len(dataloader)
    )


    # ========================================================
    # SAVE CHECKPOINTS
    # ========================================================

    torch.save(
        generator.state_dict(),
        GENERATOR_CHECKPOINT
    )

    torch.save(
        discriminator.state_dict(),
        DISCRIMINATOR_CHECKPOINT
    )


    # ========================================================
    # EPOCH SUMMARY
    # ========================================================

    print()
    print("-----------------------------------------------")
    print(
        f"Epoch {epoch + 1}/{NUM_EPOCHS} completed."
    )
    print(
        f"Average Generator Loss: "
        f"{average_generator_loss:.4f}"
    )
    print(
        f"Average Discriminator Loss: "
        f"{average_discriminator_loss:.4f}"
    )
    print("Generator checkpoint saved.")
    print("Discriminator checkpoint saved.")
    print("-----------------------------------------------")
    print()


# ============================================================
# TRAINING COMPLETE
# ============================================================

print()
print("================================================")
print("FACADES TRAINING COMPLETED SUCCESSFULLY!")
print("================================================")
print(
    f"Generator: {GENERATOR_CHECKPOINT}"
)
print(
    f"Discriminator: {DISCRIMINATOR_CHECKPOINT}"
)
print("================================================")