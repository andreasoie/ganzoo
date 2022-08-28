import math
from typing import List, Sequence, Tuple

import torch
from torch import Tensor, nn


def make_block(
    upsample: int,
    in_channels: int,
    out_channels: int,
    kernel_size: int = 3,
    stride: int = 1,
    padding: int = 1,
    bias: bool = False,
    n_power_iterations: int = 10,
):
    return nn.Sequential(
        nn.utils.spectral_norm(
            module=nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=bias,
            ),
            n_power_iterations=n_power_iterations,
        ),
        nn.Upsample(upsample),
    )


class Unsqueeze(nn.Module):
    def forward(self, x):
        return x.unsqueeze(1)


class Squeeze(nn.Module):
    def forward(self, x):
        return x.squeeze(1)


class Generator(nn.Module):
    def __init__(
        self,
        z_dim: int = None,
        upsamples: List[int] = None,
        img_shape: Tuple[int, int, int] = [1, 28, 28],
    ) -> None:
        # channels « 128, 256, 512, 1024 »
        super().__init__()
        self.l1 = nn.Sequential(
            nn.Linear(in_features=100, out_features=100),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )
        self.l2 = make_block(upsample=100, in_channels=1, out_channels=32)
        self.l3 = make_block(upsample=200, in_channels=32, out_channels=32)
        self.l4 = make_block(upsample=400, in_channels=32, out_channels=32)
        self.l5 = make_block(upsample=800, in_channels=32, out_channels=1)
        self.l6 = nn.Sequential(
            nn.Linear(in_features=800, out_features=math.prod(img_shape)),
            nn.Tanh(),
        )

        self.model = nn.Sequential(
            nn.Linear(in_features=100, out_features=100),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            Unsqueeze,
            make_block(upsample=100, in_channels=1, out_channels=32),
            make_block(upsample=200, in_channels=32, out_channels=32),
            make_block(upsample=400, in_channels=32, out_channels=32),
            make_block(upsample=800, in_channels=32, out_channels=1),
            Squeeze,
            nn.Linear(in_features=800, out_features=math.prod(img_shape)),
            nn.Tanh(),
        )

    def _forward(self, x: Tensor) -> Tensor:
        return self.model(x)

    def forward_with_print(self, x: Tensor) -> Tensor:
        print("IN =", x.shape)
        out = self.l1(x)
        print("L1 =", out.shape)
        out = out.unsqueeze(1)
        print("US =", out.shape)
        out = self.l2(out)
        print("L2 =", out.shape)
        out = self.l3(out)
        print("L3 =", out.shape)
        out = self.l4(out)
        print("L4 =", out.shape)
        out = self.l5(out)
        print("L5 =", out.shape)
        out = out.squeeze(1)
        print("SQ =", out.shape)
        out = self.l6(out)
        print("L6 =", out.shape)
        return out


class Penerator(nn.Module):
    def __init__(self, img_shape: Tuple[int, int, int] = [1, 28, 28]):
        super().__init__()
        self.img_shape = img_shape

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(100, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, math.prod(self.img_shape)),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.shape[0], *self.img_shape)
        return img


def main():

    BS = 64
    ZDIM = 100

    gen = Generator()
    pen = Penerator()

    noise = torch.randn(BS, ZDIM)

    out = gen(noise)
    out = pen(noise)


if __name__ == "__main__":
    raise SystemExit(main())
