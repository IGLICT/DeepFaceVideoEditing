import abc
import os
import os.path
import jittor as jt
import jittor.transform as transform
import argparse

from lpips.lpips import LPIPS
from training_stylegan3.projectors import w_projector
from configs import global_config, paths_config, hyperparameters
from criteria import l2_loss
from criteria.localitly_regulizer import Space_Regulizer
from models.stylegan3.networks_stylegan3 import Generator
from models.psp_stylegan3 import pSp

from utils.util import load_old_G, read_img, save_img

class BaseCoach:
    def __init__(self, data_loader, use_wandb):
        self.use_wandb = use_wandb
        self.data_loader = data_loader
        self.w_pivots = {}
        self.image_counter = 0

        if hyperparameters.first_inv_type == 'w+':
            self.initilize_e4e()

        self.e4e_image_transform = transform.Compose([
            transform.Resize((256, 256)),
            transform.ImageNormalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

        # Initialize loss
        self.lpips_loss = LPIPS(net=hyperparameters.lpips_type, spatial=False)

        self.restart_training()

    def restart_training(self):
        # Initialize networks
        self.G = load_old_G(paths_config.StyleGAN3_weights)
        self.original_G = load_old_G(paths_config.StyleGAN3_weights)
        self.optimizer = self.configure_optimizers()
    
    def get_inversion(self, w_path_dir, image_name, image):
        w_pivot = None

        #if hyperparameters.use_last_w_pivots:
        #    w_pivot = self.load_inversions(w_path_dir, image_name)

        if not hyperparameters.use_last_w_pivots or w_pivot is None:
            w_pivot = self.calc_inversions(image, image_name)

        return w_pivot

    def load_inversions(self, w_path_dir, image_name):
        if image_name in self.w_pivots:
            return self.w_pivots[image_name]
        if hyperparameters.first_inv_type == 'w+':
            w_potential_path = f'{w_path_dir}/{paths_config.e4e_results_keyword}/{image_name}/0.pt'
        else:
            w_potential_path = f'{w_path_dir}/{paths_config.pti_results_keyword}/{image_name}/0.pt'
        if not os.path.isfile(w_potential_path):
            return None
        w = jt.load(w_potential_path)
        w = jt.array(w)
        self.w_pivots[image_name] = w
        return w
    
    def calc_inversions(self, image, image_name):
        if hyperparameters.first_inv_type == 'w+':
            w = self.get_e4e_inversion(image)
        else:
            id_image = image
            w = w_projector.project(self.G, id_image,  w_avg_samples=600, num_steps=hyperparameters.first_inv_steps)
        return w

    @abc.abstractmethod
    def train(self):
        pass

    def configure_optimizers(self):
        optimizer = jt.nn.Adam(self.G.parameters(), lr=hyperparameters.pti_learning_rate)

        return optimizer
    
    def calc_loss(self, generated_images, real_images):
        loss = 0.0

        if hyperparameters.pt_l2_lambda > 0:
            l2_loss_val = l2_loss.l2_loss(generated_images, real_images)
            loss += l2_loss_val * hyperparameters.pt_l2_lambda
        
        if hyperparameters.pt_lpips_lambda > 0:
            loss_lpips = self.lpips_loss(generated_images, real_images)
            loss += loss_lpips * hyperparameters.pt_lpips_lambda

        return loss, l2_loss_val, loss_lpips

    def execute(self, w):
        generated_images = self.G.synthesis(w)

        return generated_images

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
    
