import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import DEVICE, IMAGE_SIZE
from dataset import Pix2PixDataset
from model import Generator, Discriminator


dataset = Pix2PixDataset(
    root_dir="dataset",
    image_size=IMAGE_SIZE
)

dataloader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=True,
    num_workers=0
)

generator = Generator().to(DEVICE)
discriminator = Discriminator().to(DEVICE)

adversarial_loss = nn.BCEWithLogitsLoss()
l1_loss = nn.L1Loss()

optimizer_G = torch.optim.Adam(
    generator.parameters(),
    lr=0.0002,
    betas=(0.5, 0.999)
)

optimizer_D = torch.optim.Adam(
    discriminator.parameters(),
    lr=0.0002,
    betas=(0.5, 0.999)
)

input_image, target_image = next(iter(dataloader))

input_image = input_image.to(DEVICE)
target_image = target_image.to(DEVICE)

print("Input:", input_image.shape)
print("Target:", target_image.shape)

# -------------------------
# Generator
# -------------------------

fake_image = generator(input_image)

print("Generated:", fake_image.shape)

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
    100 * reconstruction_loss
)

optimizer_G.zero_grad()
generator_loss.backward()
optimizer_G.step()

# -------------------------
# Discriminator
# -------------------------

real_prediction = discriminator(
    input_image,
    target_image
)

fake_prediction = discriminator(
    input_image,
    fake_image.detach()
)

real_loss = adversarial_loss(
    real_prediction,
    torch.ones_like(real_prediction)
)

fake_loss = adversarial_loss(
    fake_prediction,
    torch.zeros_like(fake_prediction)
)

discriminator_loss = (
    real_loss + fake_loss
) * 0.5

optimizer_D.zero_grad()
discriminator_loss.backward()
optimizer_D.step()

print("Generator loss:", generator_loss.item())
print("Discriminator loss:", discriminator_loss.item())
print("✅ One training step successful!")