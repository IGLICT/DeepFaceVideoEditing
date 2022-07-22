
import os
import os.path
import jittor as jt
import jittor.nn as nn
import jittor.transform as transform
from PIL import Image
import numpy as np
from typing import Any

from models.stylegan3.networks_stylegan3 import Generator

class EasyDict(dict):
    """Convenience class that behaves like a dict but allows access with the attribute syntax."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def __delattr__(self, name: str) -> None:
        del self[name]

def load_old_G(ckpt):
    kwargs = EasyDict(
            z_dim                   = 512,
            c_dim                   = 0,
            w_dim                   = 512,
            img_resolution          = 1024,
            img_channels            = 3,
        )
    #print(kwargs)
    G = Generator(**kwargs)
    weight_dict = jt.load(ckpt)
    G.load_state_dict(weight_dict)
    return G

def read_img(path, img_size=1024):
    #img_size = 256
    transform_image = transform.Compose([
            transform.Resize(size = img_size),
            transform.ImageNormalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    img = Image.open(path).convert('RGB')
    img = transform_image(img)
    img = jt.array(img)
    img = img.unsqueeze(0)
    return img

def save_img(image, path):
    image = image.squeeze(0).detach().numpy()
    image = (np.transpose(image, (1, 2, 0)) + 1) / 2.0 * 255.0
    image = np.clip(image, 0, 255).astype(np.uint8)
    image = Image.fromarray(image)
    image.save(path)

