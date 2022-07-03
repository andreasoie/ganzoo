from torch.utils.data import DataLoader
from torchvision import datasets, transforms

BATCH_SIZE = 64
IMAGE_SHAPE = (1, 28, 28)

train_tfms = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
)
test_tfms = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
)

train_data = datasets.MNIST(
    root="./data", train=True, download=True, transform=train_tfms
)
test_data = datasets.MNIST(
    root="./data", train=False, download=True, transform=test_tfms
)

train_loader = DataLoader(
    train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True
)
test_loader = DataLoader(
    test_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True
)


# ....
