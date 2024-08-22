
from torchvision import datasets, transforms
from torch import nn


transf = transforms.Compose([
    transforms.ToTensor()
])

train_ds = datasets.MNIST(root="dataset/train", download=True, transform=transf)
test_val_ds = datasets.MNIST(root="dataset/test_val", download=True, train=False, transform=transf)
