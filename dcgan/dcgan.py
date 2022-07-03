"""
DCAN from scratch
"""

from typing import List, Tuple

import torch


def get_channel_sizes(image_shape: Tuple[int], reverse: bool = False) -> List[int]:
    assert image_shape[1] == image_shape[2], "Image must be square"
    output_channels = [
        8 * image_shape[1],
        4 * image_shape[1],
        2 * image_shape[1],
        1 * image_shape[1],
        1 * image_shape[0],
    ]
    if reverse:
        output_channels = output_channels[::-1]
    return output_channels


class Generator(torch.nn.Module):
    """Upsamples by uses fractionally-strided convolutions"""

    def __init__(
        self,
        latent_dimensions: int,
        image_shape: Tuple[int],
    ) -> None:
        super().__init__()
        self.output_channels = get_channel_sizes(image_shape)
        self.conv1 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(
                in_channels=latent_dimensions,
                out_channels=self.output_channels[0],
                kernel_size=4,
                stride=1,
                padding=0,
                bias=False,
            ),
            torch.nn.BatchNorm2d(num_features=self.output_channels[0]),
            torch.nn.ReLU(),
        )

        self.conv2 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(
                in_channels=self.output_channels[0],
                out_channels=self.output_channels[1],
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            torch.nn.BatchNorm2d(num_features=self.output_channels[1]),
            torch.nn.ReLU(),
        )

        self.conv3 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(
                in_channels=self.output_channels[1],
                out_channels=self.output_channels[2],
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            torch.nn.BatchNorm2d(num_features=self.output_channels[2]),
            torch.nn.ReLU(),
        )

        self.conv4 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(
                in_channels=self.output_channels[2],
                out_channels=self.output_channels[3],
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            torch.nn.BatchNorm2d(num_features=self.output_channels[3]),
            torch.nn.ReLU(),
        )

        self.conv5 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(
                in_channels=self.output_channels[3],
                out_channels=self.output_channels[4],
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            torch.nn.Tanh(),
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        return out


class Discriminator(torch.nn.Module):
    """Downsampled by uses convolutions"""

    def __init__(self, image_shape: Tuple[int]) -> None:
        super().__init__()
        self.output_channels = get_channel_sizes(image_shape, reverse=True)
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=self.output_channels[0],
                out_channels=self.output_channels[1],
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            torch.nn.LeakyReLU(negative_slope=0.2),
        )

        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=self.output_channels[1],
                out_channels=self.output_channels[2],
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            torch.nn.BatchNorm2d(num_features=self.output_channels[2]),
            torch.nn.LeakyReLU(negative_slope=0.2),
        )

        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=self.output_channels[2],
                out_channels=self.output_channels[3],
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            torch.nn.BatchNorm2d(num_features=self.output_channels[3]),
            torch.nn.LeakyReLU(negative_slope=0.2),
        )

        self.conv4 = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=self.output_channels[3],
                out_channels=self.output_channels[4],
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            torch.nn.BatchNorm2d(num_features=self.output_channels[4]),
            torch.nn.LeakyReLU(negative_slope=0.2),
        )
        self.conv5 = torch.nn.Sequential(
            torch.nn.Conv2d(
                self.output_channels[4],
                out_channels=1,
                kernel_size=4,
                stride=1,
                padding=0,
                bias=False,
            ),
            torch.nn.Sigmoid(),
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        return out


class DCGAN(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        pass


if __name__ == "__main__":

    latent_dimensions = 100
    image_shape = (1, 28, 28)

    generator = Generator(latent_dimensions=latent_dimensions, image_shape=image_shape)
    disriminator = Discriminator(image_shape=image_shape)

    test_input = torch.rand(1, latent_dimensions, 1, 1)
    output = generator(test_input)
    output = disriminator(output)

    """
    when
        image shape (1, 28, 28)

    then:
        [0] =  torch.Size([1, 100, 1, 1])
        [1] =  torch.Size([1, 224, 4, 4])
        [2] =  torch.Size([1, 112, 8, 8])
        [3] =  torch.Size([1, 56, 16, 16])
        [4] =  torch.Size([1, 28, 32, 32])
        [5] =  torch.Size([1, 1, 64, 64])

        [5] =  torch.Size([1, 1, 64, 64])
        [4] =  torch.Size([1, 28, 32, 32])
        [3] =  torch.Size([1, 56, 16, 16])
        [2] =  torch.Size([1, 112, 8, 8])
        [1] =  torch.Size([1, 224, 4, 4])
        [0] =  torch.Size([1, 1, 1, 1])
    """
