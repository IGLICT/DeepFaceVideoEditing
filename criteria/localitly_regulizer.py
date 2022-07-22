import jittor as jt
import numpy as np
from criteria import l2_loss
from configs import hyperparameters
from configs import global_config

class Space_Regulizer:
    def __init__(self, original_G, lpips_net):
        self.original_G = original_G
        self.morphing_regulizer_alpha = hyperparameters.regulizer_alpha
        self.lpips_loss = lpips_net

    def get_morphed_w_code(self, new_w_code, fixed_w):
        interpolation_direction = new_w_code - fixed_w
        interpolation_direction_norm = jt.norm(interpolation_direction, p=2)
        direction_to_move = hyperparameters.regulizer_alpha * interpolation_direction / interpolation_direction_norm
        result_w = fixed_w + direction_to_move
        self.morphing_regulizer_alpha * fixed_w + (1 - self.morphing_regulizer_alpha) * new_w_code

        return result_w
    
    def get_image_from_ws(self, w_codes, G):
        return jt.concat([G([w_code], input_is_latent=True) for w_code in w_codes])
    
    def ball_holder_loss_lazy(self, new_G, num_of_sampled_latents, w_batch, use_wandb=False):
        loss = 0.0

        z_samples = np.random.randn(num_of_sampled_latents, self.original_G.style_dim)
        #truncation_psi=0.5
        #w_samples = self.original_G.get_latent(jt.array(z_samples)), truncation_psi=0.5)
        w_samples = self.original_G.get_latent(jt.array(z_samples))
        territory_indicator_ws = [self.get_morphed_w_code(w_code.unsqueeze(0), w_batch) for w_code in w_samples]

        for w_code in territory_indicator_ws:
            new_img = new_G([w_code], input_is_latent=True, randomize_noise=False)
            with jt.no_grad():
                old_img = self.original_G([w_code], input_is_latent=True, randomize_noise=False)

            if hyperparameters.regulizer_l2_lambda > 0:
                l2_loss_val = l2_loss.l2_loss(old_img, new_img)
                #if use_wandb:
                #    wandb.log({f'space_regulizer_l2_loss_val': l2_loss_val.detach().cpu()},
                #              step=global_config.training_step)
                loss += l2_loss_val * hyperparameters.regulizer_l2_lambda

            if hyperparameters.regulizer_lpips_lambda > 0:
                loss_lpips = self.lpips_loss(old_img, new_img)
                loss_lpips = torch.mean(torch.squeeze(loss_lpips))
                #if use_wandb:
                #    wandb.log({f'space_regulizer_lpips_loss_val': loss_lpips.detach().cpu()},
                #              step=global_config.training_step)
                loss += loss_lpips * hyperparameters.regulizer_lpips_lambda

        return loss / len(territory_indicator_ws)
    



