import os
import sys

import torch
import torchvision.utils as vutils
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from dcgan.model import Discriminator, Generator, initialize_weights
from tools.utils import get_device, set_seeds

torch.backends.cudnn.benchmark = True
set_seeds(1337)

BS = 1024
IMAGE_SHAPE = (1, 64, 64)  # resize 28x28 to 64x64
IMAGE_RESIZE = (64, 64)

tfms = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize(IMAGE_RESIZE),
        transforms.Normalize((0.1307,), (0.3081,)),
    ]
)
dataset = datasets.MNIST(root="./datasets", train=True, download=True, transform=tfms)
dataloader = DataLoader(dataset, batch_size=BS, shuffle=True, num_workers=2)

N_EPOCHS = 25
Z_DIMENTION = 100
LEARNING_RATE = 0.0002
DEVICE = get_device()

modelG = Generator(latent_dimensions=Z_DIMENTION, image_shape=IMAGE_SHAPE)
initialize_weights(modelG)
modelG.to(DEVICE)

modelD = Discriminator(image_shape=IMAGE_SHAPE)
initialize_weights(modelD)
modelD.to(DEVICE)

optimG = torch.optim.Adam(modelG.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
optimD = torch.optim.Adam(modelD.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
criterion = torch.nn.BCELoss()

BASE_DIR = os.path.join("dcgan", "outputs")
CKT_DIR = os.path.join(BASE_DIR, "checkpoints")
IMG_DIR = os.path.join(BASE_DIR, "images")

os.makedirs(BASE_DIR, exist_ok=True)
os.makedirs(CKT_DIR, exist_ok=True)
os.makedirs(IMG_DIR, exist_ok=True)


for epoch in range(N_EPOCHS):
    gloss = []
    dloss = []

    for idx, batch in enumerate(dataloader):
        real_images = batch[0].to(DEVICE)
        n_labels = batch[1].shape[0]

        # Discriminate with REAL
        modelD.zero_grad()
        true_labels = torch.ones((n_labels,), dtype=real_images.dtype, device=DEVICE)
        logits = modelD(real_images)
        lossD_real = criterion(logits, true_labels)
        lossD_real.backward()
        D_x = logits.mean().item()

        # Discriminate with FAKE
        noise_vector = torch.randn(n_labels, Z_DIMENTION, 1, 1, device=DEVICE)
        fake_images = modelG(noise_vector)
        fake_labels = torch.zeros((n_labels,), dtype=fake_images.dtype, device=DEVICE)
        logits = modelD(fake_images.detach())
        lossD_fake = criterion(logits, fake_labels)
        lossD_fake.backward()
        D_G_z1 = logits.mean().item()
        lossD = lossD_real + lossD_fake
        optimD.step()

        # Generate
        modelG.zero_grad()
        true_labels = torch.ones((n_labels,), dtype=real_images.dtype, device=DEVICE)
        logits = modelD(fake_images)
        lossG = criterion(logits, true_labels)
        lossG.backward()
        D_G_z2 = logits.mean().item()
        optimG.step()

        gloss.append(lossG.item())
        dloss.append(lossD.item())

        print(
            f"Epoch: {epoch}, Batch: {idx}/{len(dataloader)}, D_x: {round(D_x, 4)}, D_G_z1: {round(D_G_z1, 4)}, D_G_z2: {round(D_G_z2, 4)}, G_loss: {round(lossG.item(), 4)}, D_loss: {round(lossD.item(), 4)}"
        )

        if idx % 200 == 0:
            test_noise = torch.randn(BS, Z_DIMENTION, 1, 1, device=DEVICE)
            savedir_real = os.path.join(IMG_DIR, f"real_images.png")
            savedir_fake = os.path.join(IMG_DIR, f"fake_images_{epoch}.png")
            vutils.save_image(real_images, savedir_real, normalize=True)
            fake = modelG(test_noise)
            vutils.save_image(fake.detach(), savedir_fake, normalize=True)

    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss")
    plt.plot(gloss, label="Generator")
    plt.plot(dloss, label="Discriminator")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(IMG_DIR, f"loss_{epoch}.png"))

    gpath = os.path.join(CKT_DIR, f"modelG_epoch_{epoch}.pth")
    dpath = os.path.join(CKT_DIR, f"modelD_epoch_{epoch}.pth")
    torch.save(modelG.state_dict(), gpath)
    torch.save(modelD.state_dict(), dpath)
