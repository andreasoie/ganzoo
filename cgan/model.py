import torch
from torch import nn


class Generator(nn.Module):
    def __init__(self, img_shape: int, n_classes: int, latent_dim: int) -> None:
        super().__init__()
        self.img_shape = img_shape
        self.n_classes = n_classes
        self.latent_dim = latent_dim

        self.embedder = nn.Embedding(self.n_classes, self.n_classes)
        input_channels = self.latent_dim + self.n_classes
        output_channels = torch.tensor(self.img_shape).prod()

        self.model = nn.Sequential(
            nn.Linear(input_channels, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, output_channels, bias=False),
            nn.Tanh(),
        )

    def forward(self, noise: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        label_embedding = self.embedder(labels)
        noise = torch.cat((label_embedding, noise), -1)
        signal = self.model(noise)
        signal = signal.view(signal.size(0), *self.img_shape)
        return signal


class Discriminator(nn.Module):
    def __init__(self, img_shape: int, n_classes: int) -> None:
        super().__init__()
        self.img_shape = img_shape
        self.n_classes = n_classes

        self.embedder = nn.Embedding(self.n_classes, self.n_classes)
        input_channels = torch.tensor(self.img_shape).prod() + self.n_classes
        output_channels = 1

        self.model = nn.Sequential(
            nn.Linear(input_channels, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 128),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, output_channels),
        )

    def forward(self, img: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        label_embedding = self.embedder(labels)
        img = img.view(img.size(0), -1)  # e.g. [BS, 1, 28, 28] -> [BS, 784]
        img = torch.cat((img, label_embedding), -1)
        return self.model(img)


if __name__ == "__main__":

    N_CLASSES = 10
    IMAGE_SHAPE = 1, 28, 28

    BATCH_SIZE = 256
    LATENT_DIM = 100

    gen = Generator(IMAGE_SHAPE, N_CLASSES, LATENT_DIM)
    dis = Discriminator(IMAGE_SHAPE, N_CLASSES)

    noise_shape = (BATCH_SIZE, LATENT_DIM)
    noise = torch.randn(noise_shape)
    rand_labels = torch.randint(0, N_CLASSES, (BATCH_SIZE,))

    seq = gen(noise, rand_labels)
    probs = dis(seq, rand_labels)

    print("Noise :", noise.shape)
    print("  Seq :", seq.shape)
    print("Probs :", probs.shape)
