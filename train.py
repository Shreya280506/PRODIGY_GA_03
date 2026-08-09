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
    IMAGE_SIZE,
    CHECKPOINT_DIR
)

from dataset import Pix2PixDataset
from model import Generator, Discriminator


# Create checkpoint directory
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


# Dataset
dataset = Pix2PixDataset(
    root_dir="dataset",
    image_size=IMAGE_SIZE
)

dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)


# Models
generator = Generator().to(DEVICE)
discriminator = Discriminator().to(DEVICE)


# Loss functions
adversarial_loss = nn.BCEWithLogitsLoss()
l1_loss = nn.L1Loss()


# Optimizers
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


print(f"Using device: {DEVICE}")
print(f"Training images: {len(dataset)}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Epochs: {NUM_EPOCHS}")


# Training loop
for epoch in range(NUM_EPOCHS):

    progress_bar = tqdm(
        dataloader,
        desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}"
    )

    for input_image, target_image in progress_bar:

        input_image = input_image.to(DEVICE)
        target_image = target_image.to(DEVICE)

        # ==========================================
        # Train Generator
        # ==========================================

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


        # ==========================================
        # Train Discriminator
        # ==========================================

        optimizer_D.zero_grad()

        real_prediction = discriminator(
            input_image,
            target_image
        )

        real_loss = adversarial_loss(
            real_prediction,
            torch.ones_like(real_prediction)
        )

        fake_prediction = discriminator(
            input_image,
            fake_image.detach()
        )

        fake_loss = adversarial_loss(
            fake_prediction,
            torch.zeros_like(fake_prediction)
        )

        discriminator_loss = (
            real_loss + fake_loss
        ) * 0.5

        discriminator_loss.backward()
        optimizer_D.step()


        progress_bar.set_postfix(
            G_loss=f"{generator_loss.item():.4f}",
            D_loss=f"{discriminator_loss.item():.4f}"
        )


    # Save model after every epoch
    torch.save(
        generator.state_dict(),
        os.path.join(
            CHECKPOINT_DIR,
            "generator.pth"
        )
    )

    torch.save(
        discriminator.state_dict(),
        os.path.join(
            CHECKPOINT_DIR,
            "discriminator.pth"
        )
    )

    print(
        f"\nEpoch {epoch + 1} completed. "
        "Models saved."
    )


print("\nTraining completed successfully!")