import copy
import numpy as np
import time
from PIL import Image
import jittor as jt
from tqdm import tqdm

from configs import global_config, hyperparameters, paths_config
from lpips.lpips import LPIPS
from models.sketch_branch import SketchGenerator
from utils.util import load_old_G, read_img, save_img

def project(
        G,
        target,
        target_sketch,
        target_mask,
        initial_w,
        num_steps=500,
        initial_learning_rate=0.01,
        regularize_noise_weight=1e5,
):

    def logprint(*args):
        if verbose:
            print(*args)

    G.eval()

    stylegan_sketch = SketchGenerator(img_resolution = 1024, sketch_channel=32, output_nc = 3)
    save_path = paths_config.sketch_branch_weights
    pretrained_dict = jt.load(save_path)
    stylegan_sketch.load_state_dict(pretrained_dict)
    stylegan_sketch.eval()

    # Features for target image.
    target_images = target
    if target_images.shape[2] > 256:
        target_images = jt.nn.interpolate(target_images, size=(256, 256), mode='bilinear')

    # Preprocess for mask
    target_mask_256 = jt.nn.interpolate(target_mask, size=(256, 256), mode='bilinear')
    target_no_mask_256 = target_mask_256 < 0.0
    target_images = target_images * target_no_mask_256
    target_mask = target_mask > 0.0
    target_sketch = target_sketch.detach() * target_mask.detach()

    # Get w_opt with initial_w
    w_opt = jt.array(initial_w).detach()  # pylint: disable=not-callable
    
    optimizer = jt.nn.Adam([w_opt], betas=(0.9, 0.999),
                                 lr=hyperparameters.first_inv_lr)
    loss_fn = LPIPS(net='alex', spatial=False)
    loss_sketch = LPIPS(net='alex', spatial=False)

    for step in tqdm(range(num_steps)):
        feature_list, synth_images = G.synthesis.execute_lists(w_opt)

        # calculate sketch loss
        lambda_sketch_stylegan = hyperparameters.sketch_weight
        if lambda_sketch_stylegan != 0.0:
            sketch_stylegan = stylegan_sketch(feature_list)
            sketch_stylegan = sketch_stylegan * target_mask.detach()

            sketch_stylegan_loss = loss_sketch(target_sketch, sketch_stylegan) * lambda_sketch_stylegan
        else:
            sketch_stylegan_loss = 0.0
        
        # calculate image loss
        lambda_dist = hyperparameters.image_weight
        if lambda_dist != 0.0:
            # Downsample image to 256x256 if it's larger than that. VGG was built for 224x224 images.
            if synth_images.shape[2] > 256:
                synth_images = jt.nn.interpolate(synth_images, size=(256, 256), mode='bilinear')
                synth_images = target_no_mask_256.detach() * synth_images
            # Features for synth images.
            dist = loss_fn(target_images, synth_images) * lambda_dist
        else:
            dist = 0.0
        
        loss = dist + sketch_stylegan_loss

        print(f'step {step + 1:>4d}/{num_steps}: dist {dist:<4.2f} sketch_stylegan_loss {float(sketch_stylegan_loss):<5.2f} loss {float(loss):<5.2f}')
        optimizer.step(loss)
    
    return w_opt

