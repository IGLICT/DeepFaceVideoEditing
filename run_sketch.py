import os
import jittor as jt
import jittor.nn as nn
import jittor.transform as transform
from typing import Any
from PIL import Image
import numpy as np
import argparse

from configs import paths_config, hyperparameters, global_config
from models.scp import scp
from models.psp_stylegan3 import pSp
from models.stylegan3.networks_stylegan3 import Generator
from training_stylegan3.projectors import w_projector_sketch
from utils.util import load_old_G, read_img, save_img

jt.flags.use_cuda = 1

class SoftDilate(nn.Module):
    def __init__(self, kernel_size=15, threshold=0.6, iterations=1):
        super(SoftDilate, self).__init__()
        r = kernel_size // 2
        self.padding1 = (r, r, r, r)
        self.padding = r
        self.iterations = iterations
        self.threshold = threshold

        # Create kernel
        y_indices, x_indices = jt.misc.meshgrid(jt.misc.arange(0., kernel_size), jt.misc.arange(0., kernel_size))
        dist = jt.sqrt((x_indices - r) ** 2 + (y_indices - r) ** 2)
        kernel = dist.max() - dist
        kernel /= kernel.sum()
        kernel = kernel.view(1, 1, *kernel.shape)
        self.weight = kernel

    def execute(self, x_ori):
        x = 1.0 - x_ori
        x = x.float()
        for i in range(self.iterations - 1):
            midx = nn.pad(x, self.padding1, mode="reflect")
            midx = nn.conv2d(midx, weight=self.weight, groups=x.shape[1], padding=0)
            #print(midx.shape)
            #print(x.shape)
            x = jt.minimum(x, midx)
        x = nn.pad(x, self.padding1, mode="reflect")
        x = nn.conv2d(x, weight=self.weight, groups=x.shape[1], padding=0)

        x = 1.0 - x
        y = x.clone()

        mask = x >= self.threshold
        x[mask] = 1.0
        mask_not = jt.logical_not(mask)
        x[mask_not] /= x[mask_not].max()

        return x, y

class E4E:
    def __init__(self):
        self.initilize_e4e()

        self.e4e_image_transform = transform.Compose([
            transform.Resize((256, 256)),
            transform.ImageNormalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
    
    def initilize_e4e(self):
        ckpt = jt.load(paths_config.e4e)
        opts = ckpt['opts']
        opts['checkpoint_path'] = paths_config.e4e
        opts = argparse.Namespace(**opts)

        self.e4e_inversion_net = pSp(opts)
        self.e4e_inversion_net.eval()
        self.e4e_inversion_net.requires_grad_ = False

    def get_e4e_inversion(self, image):
        new_image = jt.nn.interpolate(image, size=(256, 256), mode='bilinear')
        _, w = self.e4e_inversion_net(new_image, randomize_noise=False, return_latents=True, resize=False,
                                      input_code=False)
        return w

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--edit_sketch_path", type=str, default = "edit/baseShape/edit1/", help = "the path of edit sketch")
    parser.add_argument("--sketch_weight", type=float, default = 20.0, help = "the weight of sketch loss")
    parser.add_argument("--image_weight", type=float, default = 20.0, help = "the weight of image loss")
    parser.add_argument("--optimize_steps", type=int, default = 100, help = "the steps of optimization")
    parser.add_argument("--use_scp", type=bool, default = True, help = "use scp model or not")
    args = parser.parse_args()

    edit_path = os.path.join(paths_config.input_video_path, args.edit_sketch_path)
    mask = read_img(os.path.join(edit_path, 'mask_edit.jpg'), 512)
    sketch = read_img(os.path.join(edit_path, 'sketch_edit.jpg'), 512)
    img = read_img(os.path.join(edit_path, 'img.jpg'), 256)

    #################################################################################
    #Generate initial_w
    initial_w_path = os.path.join(edit_path, 'initial_w.pkl')
    #if os.path.exists(initial_w_path):
    #    initial_w = jt.array(jt.load(initial_w_path))
    #else:
    checkpoint_path = paths_config.scp_weights
    ckpt = jt.load(checkpoint_path)
    opts = ckpt['opts']
    opts['checkpoint_path'] = checkpoint_path
    opts = argparse.Namespace(**opts)
    net = scp(opts)
    net.eval()
    
    if args.use_scp:
        sketch_scp = jt.nn.interpolate(sketch, [256, 256])
        sketch_scp = sketch_scp[:,0:1,:,:]
        mask_scp = jt.nn.interpolate(mask, [256, 256])
        mask_scp = mask_scp[:,0:1,:,:]
        mask_scp = mask_scp > 0.5
        mask_scp = jt.ones(sketch_scp.shape) * mask_scp
        sketch_scp = sketch_scp * mask_scp
        img_scp = img * (1 - mask_scp)  
        _, initial_w, _ = net(img_scp, sketch_scp, mask_scp, resize = False)
        del net, ckpt, sketch_scp, mask_scp, img_scp
    else:
        initial_w = net.img_encoder(img)
        del net
    jt.save(initial_w, initial_w_path)
    
    #################################################################################
    #Optimize sketch to generate editing vectors
    initial_w = initial_w.detach()
    weight_G_path = os.path.join(paths_config.input_video_path, 'ffhq_weights_stylegan3.pkl')
    G = load_old_G(weight_G_path)

    #w_noise = jt.randn_like(initial_w) * hyperparameters.w_noise_scale
    #initial_w = initial_w + w_noise

    w_refine = w_projector_sketch.project(G, target=img, target_sketch=sketch, target_mask=mask, initial_w=initial_w, \
                                             sketch_weight=args.sketch_weight, image_weight=args.image_weight, num_steps=args.optimize_steps)
    img_refine = G.synthesis(w_refine)
    #save_img(img_refine, os.path.join(edit_path, 'refine.jpg'))

    save_path = os.path.join(edit_path, 'refine_w.pkl')
    jt.save(w_refine, save_path)

    ##################################################################################
    #Generate mask fusion results for single editing manipulation
    ###########[0   1   2   3   4   5   6    7    8    9    10   11    12    13    #14   #15 ]
    w_refine = w_refine.detach()
    res_list = [16, 16, 16, 32, 32, 64, 128, 128, 256, 256, 512, 1024, 1024, 1024, 1024, 1024]
    smooth_mask_class = SoftDilate(kernel_size=15, threshold=0.9, iterations=10)
    coach = E4E()

    w_before = coach.get_e4e_inversion(img)

    x_lists = []
    mask_list = []
    x_edit_list, img = G.synthesis.execute_lists(w_refine, noise_mode='const')
    x_lists.append(x_edit_list)
    mask_edit = mask
    mask_edit = mask_edit[:, 0:1,:,:]
    mask_edit = (mask_edit + 1.0) / 2.0
    mask_edit, _ = smooth_mask_class(mask_edit)
    mask_list.append(mask_edit)
    
    start_layer = hyperparameters.start_layer
    end_layer = hyperparameters.end_layer
    generated_images = G.synthesis.execute_mask(w_before, mask_list, x_lists, start_layer, end_layer, noise_mode='const', force_fp32=True)
    resultPath = os.path.join(edit_path, 'mask_fusion_result.jpg')
    save_img(generated_images, resultPath)


