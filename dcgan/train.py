import argparse
import os

import torch
import torchvision.utils as vutils

from dcgan.model import Discriminator, Generator, initialize_weights
from tools.data import load_mnist
from tools.utils import get_device, init_outdir, set_seeds

torch.backends.cudnn.benchmark = True
set_seeds(3407)

outdir = init_outdir(__file__, "out")
parser = argparse.ArgumentParser()
parser.add_argument("--bs", type=int, default=128)
parser.add_argument("--epochs", type=int, default=8)
parser.add_argument("--lr", type=float, default=0.0002)
parser.add_argument("--zdim", type=int, default=100)
args = parser.parse_args()

image_shape = 1, 64, 64  # upscaled
dataloader = load_mnist(args.bs, image_shape)
device = get_device()

generator = Generator(args.zdim, image_shape)
initialize_weights(generator)
generator.to(device)

discriminator = Discriminator(image_shape)
initialize_weights(discriminator)
discriminator.to(device)

criterion = torch.nn.BCELoss()
optim_G = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
optim_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))

for epoch in range(args.epochs):
    print("")
    for idx, batch in enumerate(dataloader):
        real_images = batch[0].to(device)
        batch_size = batch[0].shape[0]

        noise_shape = (batch_size, args.zdim, 1, 1)
        noise = torch.randn(noise_shape, dtype=real_images.dtype, device=device)
        true_labels = torch.ones((batch_size,), dtype=real_images.dtype, device=device)
        fake_labels = torch.zeros((batch_size,), dtype=real_images.dtype, device=device)

        # Discriminate on real images
        discriminator.zero_grad()
        logits = discriminator(real_images)
        loss_real_D = criterion(logits, true_labels)
        loss_real_D.backward()
        D_x = logits.mean().item()

        # Discriminate on fake images
        fake_images = generator(noise)
        logits = discriminator(fake_images.detach())
        lossD_fake = criterion(logits, fake_labels)
        lossD_fake.backward()
        D_G_z1 = logits.mean().item()
        lossD = loss_real_D + lossD_fake
        optim_D.step()

        # Generate fake images
        generator.zero_grad()
        logits = discriminator(fake_images)
        lossG = criterion(logits, true_labels)
        lossG.backward()
        D_G_z2 = logits.mean().item()
        optim_G.step()

        if idx % (len(dataloader) // 10) == 0:
            fill, pad = " ", 4
            print(
                f"Epoch: {epoch:{fill}{pad}}",
                f"Batch: {idx:{fill}{pad}}/{len(dataloader)}",
                f"D(x): {D_x:.03f}",
                f"D(G(z1)): {D_G_z1:.03f}",
                f"D(G(z2)): {D_G_z2:.03f}",
                f"G_loss: {lossG:.03f}",
                f"D_loss: {lossD:.03f}",
                sep=" | ",
            )
    # ---
    NUMBERS_TO_SHOW = 16
    dummy_shape = (NUMBERS_TO_SHOW, args.zdim, 1, 1)
    dummynoise = torch.randn(dummy_shape, dtype=real_images.dtype, device=device)
    fake = generator(dummynoise)
    path_real = os.path.join(outdir, f"real_0.png")
    path_fake = os.path.join(outdir, f"fake_{epoch}.png")
    vutils.save_image(fake.detach(), path_fake, normalize=True)
    vutils.save_image(
        real_images[0:NUMBERS_TO_SHOW, :, :, :], path_real, normalize=True
    )
