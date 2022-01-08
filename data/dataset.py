from paddle.io import Dataset
from paddle.vision.transforms import RandomResizedCrop, Resize, ToTensor, Normalize, Compose

from pathlib import Path
from PIL import Image
import numpy as np

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


def recover_normalized_image(img, mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD):
    mean = np.array(mean, dtype='float32').reshape([3, 1, 1]) * 255
    std = np.array(std, dtype='float32').reshape([3, 1, 1]) * 255
    new_img = img * std + mean

    return new_img.astype('uint8')


def show_image(imgs):
    import matplotlib.pyplot as plt
    for i in range(len(imgs)):
        plt.subplot(1, len(imgs), i+1)
        plt.imshow(imgs[i].transpose(1, 2, 0))
    plt.show()


class ImageNet2012(Dataset):
    def __init__(self, data_dir, data_type, input_size=224, transform=None):
        super().__init__()
        self.data_dir = Path(data_dir) / data_type
        self.img_dir = self.data_dir / 'images'
        self.anno_file = self.data_dir / 'annotations.txt'
        self.img_names = []
        self.prepare_all_img_names()
        self.input_size = input_size
        if transform is not None:
            self._transform = transform
        else:
            self._transform = Compose([
                RandomResizedCrop(224),
                ToTensor(),
                Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
            ])

    def prepare_all_img_names(self):
        with open(self.anno_file) as f:
            for l in f:
                img_name, label = l.strip().split(' ')
                self.img_names.append((img_name, int(label)))

    def __getitem__(self, item):
        img_name, target = self.img_names[item]
        img_path = self.img_dir / img_name
        image = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        return image, target

    def __len__(self):
        return len(self.img_names)

    @property
    def transform(self):
        return self._transform

    @transform.setter
    def transform(self, t):
        self._transform = t


def create_dataset(data_dir, data_type, input_size=224):
    if data_type == 'train':
        transform = Compose([
            RandomResizedCrop(224),
            ToTensor(),
            Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
        ])
    else:
        transform = Compose([
            Resize(224),
            ToTensor(),
            Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
        ])

    return ImageNet2012(data_dir, data_type, input_size=input_size, transform=transform)
