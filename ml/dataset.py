import torchvision
from torchvision import transforms
from PIL import Image
from pathlib import Path
import os 
class MNIST:
    def __init__(self, root, transform):
        if root == None:
            root = Path('data').resolve()
            print(root)

        self.train_data = torchvision.datasets.MNIST(root=root, train=True, transform=transform, download=True)
        self.test_data = torchvision.datasets.MNIST(root=root, train=False, transform=transform, download=True)

        self.num_classes = len(self.train_data.classes)
        
class FashionMNIST:
    def __init__(self, root, transform):
        if root == None:
            root = Path('data').resolve()
            print(root)

        self.train_data = torchvision.datasets.FashionMNIST(root=root, train=True, transform=transform, download=True)
        self.test_data = torchvision.datasets.FashionMNIST(root=root, train=False, transform=transform, download=True)

        self.num_classes = len(self.train_data.classes)

class CIFAR10:
    def __init__(self, root, transform):

        if root == None:
            root = Path('data').resolve()
            print(root)

        self.train_data = torchvision.datasets.CIFAR10(root=root, train=True, transform=transform, download=True)
        self.test_data = torchvision.datasets.CIFAR10(root=root, train=False, transform=transform, download=True)

        self.num_classes = len(self.train_data.classes)

class BaseAugmentation:
    def __init__(self, resize=[32,32], mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5), **args):

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(resize, Image.BILINEAR),
            #transforms.Normalize(mean=mean, std=std),
        ])
    
    def __call__(self, image):
        return self.transform(image)
