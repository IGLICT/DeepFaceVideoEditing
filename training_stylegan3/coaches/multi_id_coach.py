import os
import jittor as jt
from tqdm import tqdm

from configs import paths_config, hyperparameters, global_config
from training_stylegan3.coaches.base_coach import BaseCoach
from utils.ImagesDataset import save_image

class MultiIDCoach(BaseCoach):
    def __init__(self, data_loader, use_wandb):
        super().__init__(data_loader, use_wandb)

    def train(self):
        self.G.synthesis.train()
        self.G.mapping.train()

        use_ball_holder = True
        w_pivots = []
        images = []

        # Calculate the w_pivots
        for fname, image in self.data_loader:
            if self.image_counter >= hyperparameters.max_images_to_invert:
                break
            image_name = fname[0]
            print('\r PTI generate w_pivot:'+image_name, end="")
            #with jt.no_grad():
            w_pivot = self.get_inversion(None, image_name, image)
            w_pivots.append(w_pivot)
            images.append((image_name, image))
            self.image_counter += 1
        print("")
        
        # Finetune the StyleGAN network
        for i in range(hyperparameters.max_pti_steps):
            self.image_counter = 0
            
            for data, w_pivot in zip(images, w_pivots):
                print('\r PTI training epochs:%d, frames:%4d'%(i,self.image_counter), end="")
                image_name, image = data

                if self.image_counter >= hyperparameters.max_images_to_invert:
                    break

                real_images_batch = image.detach()

                generated_images = self.execute(w_pivot.detach())
                loss, l2_loss_val, loss_lpips = self.calc_loss(generated_images, real_images_batch)

                self.optimizer.step(loss)

                use_ball_holder = global_config.training_step % hyperparameters.locality_regularization_interval == 0

                global_config.training_step += 1
                self.image_counter += 1
        print("")
        
        # Save PTI finetuning weights
        model_path = os.path.join(paths_config.input_video_path, 'ffhq_weights_stylegan3.pkl')
        jt.save(self.G.state_dict(), model_path)
        
        # Save PTI finetuning image results
        PTI_Path = os.path.join(paths_config.input_video_path, 'pti_results')
        if not os.path.exists(PTI_Path):
            os.makedirs(PTI_Path)
        
        image = self.execute(w_pivots[0])
        save_image(os.path.join(PTI_Path, 'after_pti.jpg'), image)
        generated_images = self.original_G.synthesis(w_pivots[0])
        save_image(os.path.join(PTI_Path, 'before_pti.jpg'), generated_images)
        image_name, image = images[0]
        save_image(os.path.join(PTI_Path, 'original_image.jpg'), image)

