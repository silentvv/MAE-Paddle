from paddle.io import Dataset
import paddle

from pathlib import Path
from PIL import Image
import numpy as np

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


class FitToFixSize:
    def __init__(self, target_size=(224, 224)):
        self.target_size = target_size

    def __call__(self, pil_img: Image, target):
        height, width = pil_img.height, pil_img.width
        if height < self.target_size[0] or width < self.target_size[0]:
            width, height = pil_img.size

            img_scale_y = self.target_size[0] / height
            img_scale_x = self.target_size[1] / width
            img_scale = min(img_scale_y, img_scale_x)
            scaled_h = int(height * img_scale)
            scaled_w = int(width * img_scale)

            new_img = Image.new("RGB", (self.target_size[1], self.target_size[0]), color=(0, 0, 0))
            img = pil_img.resize((scaled_w, scaled_h), Image.BILINEAR)
            new_img.paste(img)
        else:
            y0 = (height - self.target_size[0]) // 2
            x0 = (width - self.target_size[1]) // 2
            y1 = y0 + self.target_size[0]
            x1 = x0 + self.target_size[1]
            new_img = pil_img.crop((x0, y0, x1, y1))

        return new_img, target


class ImageToNumpy:
    def __call__(self, pil_img, target):
        np_img = np.array(pil_img, dtype=np.uint8)
        if np_img.ndim < 3:
            np_img = np.expand_dims(np_img, axis=-1)
        np_img = np.moveaxis(np_img, 2, 0)  # HWC to CHW
        return np_img, target


class ImageToTensor:
    def __init__(self, dtype=paddle.float32):
        self.dtype = dtype

    def __call__(self, pil_img, target):
        np_img = np.array(pil_img, dtype=np.uint8)
        if np_img.ndim < 3:
            np_img = np.expand_dims(np_img, axis=-1)
        np_img = np.moveaxis(np_img, 2, 0)  # HWC to CHW
        return paddle.to_tensor(np_img, dtype=self.dtype), target


class Normalize:
    def __init__(self, mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD):
        self.mean = np.array(mean, dtype='float32').reshape([3, 1, 1]) * 255
        self.std = np.array(std, dtype='float32').reshape([3, 1, 1]) * 255

    def __call__(self, image, target):
        new_image = (image - self.mean) / self.std

        return new_image, target


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


class Compose:
    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, img, annotations: dict):
        for t in self.transforms:
            img, annotations = t(img, annotations)
        return img, annotations


class ImageNet2012(Dataset):
    def __init__(self, data_dir, data_type, input_size=224, transform=None):
        super().__init__()
        self.data_dir = Path(data_dir) / data_type
        self.img_dir = self.data_dir / 'ILSVRC2012_img_val'
        self.anno_file = self.data_dir / 'val.txt'
        self.img_names = []
        self.prepare_all_img_names()
        self.input_size = input_size
        if transform is not None:
            self._transform = transform
        else:
            self._transform = Compose([
                FitToFixSize(target_size=(224, 224)),
                ImageToNumpy(),
                Normalize(),
            ])

    def prepare_all_img_names(self):
        with open(self.anno_file) as f:
            for l in f:
                img_name, label = l.strip().split(' ')
                self.img_names.append((img_name, int(label)))

    def __getitem__(self, item):
        img_name, target = self.img_names[item]
        img_path = self.img_dir / img_name
        image = Image.open(img_path)
        if self.transform is not None:
            image, target = self.transform(image, target)

        return image, target

    def __len__(self):
        return len(self.img_names)

    @property
    def transform(self):
        return self._transform

    @transform.setter
    def transform(self, t):
        self._transform = t
