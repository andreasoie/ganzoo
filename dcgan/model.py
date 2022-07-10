from typing import List, Tuple

import torch
from torch import nn


def initialize_weights(
    model: torch.nn.Module, mean: float = 0, std: float = 0.02
) -> None:
    for module in model.modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.ConvTranspose2d)):
            module.weight.data.normal_(mean=mean, std=std)
        elif isinstance(module, torch.nn.BatchNorm2d):
            module.weight.data.normal_(mean=1.0, std=std)
            module.bias.data.zero_()


def _get_channel_sizes(image_shape: Tuple[int], reverse: bool = False) -> List[int]:
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
    def __init__(self, latent_dimensions: int, image_shape: Tuple[int]) -> None:
        super().__init__()
        self.output_channels = _get_channel_sizes(image_shape)
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
            torch.nn.ReLU(inplace=True),
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
            torch.nn.ReLU(inplace=True),
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
            torch.nn.ReLU(inplace=True),
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
            torch.nn.ReLU(inplace=True),
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        return out


class Discriminator(torch.nn.Module):
    def __init__(self, image_shape: Tuple[int]) -> None:
        super().__init__()
        self.output_channels = _get_channel_sizes(image_shape, reverse=True)
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=self.output_channels[0],
                out_channels=self.output_channels[1],
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            torch.nn.LeakyReLU(negative_slope=0.2, inplace=True),
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
            torch.nn.LeakyReLU(negative_slope=0.2, inplace=True),
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
            torch.nn.LeakyReLU(negative_slope=0.2, inplace=True),
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
            torch.nn.LeakyReLU(negative_slope=0.2, inplace=True),
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = out.view(-1, 1).squeeze(1)
        return out
