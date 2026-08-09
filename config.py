import torch

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Image settings
IMAGE_SIZE = 256
CHANNELS = 3

# Training settings
BATCH_SIZE = 1
LEARNING_RATE = 0.0002
NUM_EPOCHS = 10

# Pix2Pix parameters
BETA1 = 0.5
BETA2 = 0.999
LAMBDA_L1 = 100

# Dataset and output paths
DATASET_DIR = "dataset"
CHECKPOINT_DIR = "checkpoints"
OUTPUT_DIR = "outputs"

# Model checkpoint
GENERATOR_CHECKPOINT = f"{CHECKPOINT_DIR}/generator.pth"
DISCRIMINATOR_CHECKPOINT = f"{CHECKPOINT_DIR}/discriminator.pth"

print(f"Using device: {DEVICE}")