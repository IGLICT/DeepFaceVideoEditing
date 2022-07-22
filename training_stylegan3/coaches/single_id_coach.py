import os
import jittor as jt
from tqdm import tqdm
from configs import paths_config, hyperparameters, global_config
from training_stylegan3.coaches.base_coach import BaseCoach
from utils.ImagesDataset import save_image

class SingleIDCoach(BaseCoach):

    def __init__(self, data_loader, use_wandb):
        super().__init__(data_loader, use_wandb)

    def train(self):

        w_path_dir = f'{paths_config.embedding_base_dir}/{paths_config.input_data_id}'
        os.makedirs(w_path_dir, exist_ok=True)
        os.makedirs(f'{w_path_dir}/{paths_config.pti_results_keyword}', exist_ok=True)

        use_ball_holder = True

        for fname, image in tqdm(self.data_loader):
            image_name = fname[0]

            self.restart_training()

            if self.image_counter >= hyperparameters.max_images_to_invert:
                break

            embedding_dir = f'{w_path_dir}/{paths_config.pti_results_keyword}/{image_name}'
            os.makedirs(embedding_dir, exist_ok=True)

            w_pivot = None

            if hyperparameters.use_last_w_pivots:
                #print("load inversions")
                w_pivot = self.load_inversions(w_path_dir, image_name)

            elif not hyperparameters.use_last_w_pivots or w_pivot is None:
                w_pivot = self.calc_inversions(image, image_name)

            w_pivot = w_pivot.detach()

            jt.save(w_pivot, f'{embedding_dir}/0.pt')
            log_images_counter = 0
            real_images_batch = image

            generated_images = self.execute(w_pivot)
            save_image(f'{embedding_dir}/original_G.jpg', generated_images)

            for i in tqdm(range(hyperparameters.max_pti_steps)):

                generated_images = self.execute(w_pivot)
                loss, l2_loss_val, loss_lpips = self.calc_loss(generated_images, real_images_batch, image_name,
                                                               self.G, use_ball_holder, w_pivot)

                if loss_lpips <= hyperparameters.LPIPS_value_threshold:
                    break

                self.optimizer.step(loss)

                global_config.training_step += 1
                log_images_counter += 1

            self.image_counter += 1

            jt.save(self.G,
                       f'{paths_config.checkpoints_dir}/model_{global_config.run_name}_{image_name}.pt')
            
            generated_images = self.execute(w_pivot)
            save_image(f'{embedding_dir}/recon.jpg', generated_images)
            save_image(f'{embedding_dir}/original.jpg', real_images_batch)


