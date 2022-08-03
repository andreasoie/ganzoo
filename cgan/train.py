import argparse

import torch
import torchvision.utils as vutils

from cgan.model import Discriminator, Generator
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

n_classes = 10
image_shape = 1, 64, 64  # upscale
dataloader = load_mnist(args.bs, image_shape)
device = get_device()

generator = Generator(image_shape, n_classes, args.zdim)
generator.to(device)

discriminator = Discriminator(image_shape, n_classes)
discriminator.to(device)

criterion = torch.nn.MSELoss()
optim_G = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
optim_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))

for epoch in range(args.epochs):
    for idx, batch in enumerate(dataloader):
        global_steps = epoch * len(dataloader) + idx

        real_images = batch[0].to(device)
        real_labels = batch[1].to(device)
        batch_size = real_images.shape[0]

        true_targets = torch.ones((batch_size, 1), device=device)
        fake_targets = torch.zeros((batch_size, 1), device=device)

        noise_shape = (batch_size, args.zdim)
        rand_noise = torch.randn(noise_shape, device=device)
        rand_targets = torch.randint(0, n_classes, (batch_size,), device=device)

        # Generate fake images
        fake_images = generator(rand_noise, rand_targets)
        logits = discriminator(fake_images, rand_targets)
        lossG = criterion(logits, true_targets)

        optim_G.zero_grad()
        lossG.backward()
        optim_G.step()

        # Discriminate on real images
        logits = discriminator(real_images, real_labels)
        lossD_real = criterion(logits, true_targets)
        # Discriminate on fake images
        logits = discriminator(fake_images.detach(), rand_targets)
        lossD_fake = criterion(logits, fake_targets)

        lossD = (lossD_real + lossD_fake) / 2
        optim_D.zero_grad()
        lossD.backward()
        optim_D.step()

        if idx % (len(dataloader) // 10) == 0:
            fill, pad = " ", 4
            print(
                f"Epoch: {epoch+1:{fill}{pad}}",
                f"Batch: {idx:{fill}{pad}}/{len(dataloader)}",
                f"G_loss: {lossG:.03f}",
                f"D_loss: {lossD:.03f}",
                sep=" | ",
            )

        if global_steps % 500 == 0:
            with torch.no_grad():
                generator.eval()
                targets = torch.arange(n_classes).to(device)
                noise = torch.randn((n_classes, args.zdim)).to(device)
                images = generator(noise, targets)

                _filename = f"{outdir}/{global_steps}.png"
                vutils.save_image(images, _filename, normalize=True, nrow=n_classes)
                generator.train()
