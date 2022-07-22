import os

from jittor.dataset import Dataset
from PIL import Image
import numpy as np
from imageio import imread, imsave

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tiff'
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                fname = fname.split('.')[0]
                images.append((fname, path))
    return images

def save_image(path, image):
    image = image.squeeze(0).detach().numpy()
    image = (np.transpose(image, (1, 2, 0)) + 1) / 2.0 * 255.0
    image = np.clip(image, 0, 255).astype(np.uint8)
    imsave(path, image)

class ImagesDataset(Dataset):

    def __init__(self, source_root, source_transform=None):
        super().__init__()
        self.source_paths = sorted(make_dataset(source_root))
        self.source_transform = source_transform
        self.set_attrs(total_len=len(self.source_paths))

    #def __len__(self):
    #    return len(self.source_paths)

    def __getitem__(self, index):
        fname, from_path = self.source_paths[index]
        from_im = Image.open(from_path).convert('RGB')

        if self.source_transform:
            from_im = self.source_transform(from_im)

        return fname, from_im
