from typing import Tuple

from torch.utils.data import ConcatDataset, DataLoader
from torchvision import datasets, transforms


def load_mnist(
    batch_size: int, img_shape: Tuple[int, int, int], num_workers: int = 2
) -> DataLoader:
    img_resize = img_shape[1:]
    tfms = transforms.Compose(
        [
            transforms.Resize(img_resize),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
            # using default (0.1307,), (0.3081,) is way worse (why?)
        ]
    )
    d1 = datasets.MNIST(root="./datasets", train=True, download=True, transform=tfms)
    d2 = datasets.MNIST(root="./datasets", train=False, download=True, transform=tfms)
    dataset = ConcatDataset([d1, d2])
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    return dataloader
