import jittor as jt
import jittor.transform as transform
from model import BiSeNet

import cv2
import os
import numpy as np
from skimage.filters import gaussian
from PIL import Image
from argparse import ArgumentParser

jt.flags.use_cuda = 1

def sharpen(img):
    img = img * 1.0
    gauss_out = gaussian(img, sigma=5, multichannel=True)

    alpha = 1.5
    img_out = (img - gauss_out) * alpha + img

    img_out = img_out / 255.0

    mask_1 = img_out < 0
    mask_2 = img_out > 1

    img_out = img_out * (1 - mask_1)
    img_out = img_out * (1 - mask_2) + mask_2
    img_out = np.clip(img_out, 0, 1)
    img_out = img_out * 255
    return np.array(img_out, dtype=np.uint8)


def hair(image, parsing, part=17, color=[230, 50, 20]):
    b, g, r = color      #[10, 50, 250]       # [10, 250, 10]
    tar_color = np.zeros_like(image)
    tar_color[:, :, 0] = b
    tar_color[:, :, 1] = g
    tar_color[:, :, 2] = r

    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    tar_hsv = cv2.cvtColor(tar_color, cv2.COLOR_BGR2HSV)

    if part == 12 or part == 13:
        image_hsv[:, :, 0:2] = tar_hsv[:, :, 0:2]
    else:
        image_hsv[:, :, 0:1] = tar_hsv[:, :, 0:1]

    changed = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2BGR)

    if part == 17:
        changed = sharpen(changed)

    changed[parsing != part] = image[parsing != part]
    # changed = cv2.resize(changed, (512, 512))
    return changed

#
# def lip(image, parsing, part=17, color=[230, 50, 20]):
#     b, g, r = color      #[10, 50, 250]       # [10, 250, 10]
#     tar_color = np.zeros_like(image)
#     tar_color[:, :, 0] = b
#     tar_color[:, :, 1] = g
#     tar_color[:, :, 2] = r
#
#     image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
#     il, ia, ib = cv2.split(image_lab)
#
#     tar_lab = cv2.cvtColor(tar_color, cv2.COLOR_BGR2Lab)
#     tl, ta, tb = cv2.split(tar_lab)
#
#     image_lab[:, :, 0] = np.clip(il - np.mean(il) + tl, 0, 100)
#     image_lab[:, :, 1] = np.clip(ia - np.mean(ia) + ta, -127, 128)
#     image_lab[:, :, 2] = np.clip(ib - np.mean(ib) + tb, -127, 128)
#
#
#     changed = cv2.cvtColor(image_lab, cv2.COLOR_Lab2BGR)
#
#     if part == 17:
#         changed = sharpen(changed)
#
#     changed[parsing != part] = image[parsing != part]
#     # changed = cv2.resize(changed, (512, 512))
#     return changed


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--checkpoint", default='./checkpoints/79999_iter.pkl', help="path to checkpoint to restore")
    parser.add_argument("--input_image", default='./img/input/00110.jpg', help="path to source images")
    parser.add_argument("--output_image", default='./img/makeup/00110.jpg', help="path to face parsing images")

    opt = parser.parse_args()

    # 1  face
    # 10 nose
    # 11 teeth
    # 12 upper lip
    # 13 lower lip
    # 17 hair
    num = 116
    table = {
        'hair': 17,
        'upper_lip': 12,
        'lower_lip': 13
    }
    image_path = opt.input_image

    image = cv2.imread(image_path)
    ori = image.copy()

    n_classes = 19
    net = BiSeNet(n_classes=n_classes)
    net.load_state_dict(jt.load(opt.checkpoint))
    net.eval()

    transform_image = transform.Compose([
        transform.ImageNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    with jt.no_grad():
        img = Image.open(image_path)
        img = img.resize((512, 512), Image.BILINEAR)
        img = jt.array(transform_image(img))
        img = jt.unsqueeze(img, 0)
        out = net(img)[0]
        parsing = out.squeeze(0).numpy().argmax(0)

    parsing = cv2.resize(parsing, image.shape[0:2], interpolation=cv2.INTER_NEAREST)

    parts = [table['hair'], table['upper_lip'], table['lower_lip']]
    #colors = [[20, 20, 200], [100, 100, 230], [100, 100, 230]]
    colors = [[200, 100, 100]]
    for part, color in zip(parts, colors):
        image = hair(image, parsing, part, color)
    cv2.imwrite(opt.output_image, cv2.resize(image, (512, 512)))

