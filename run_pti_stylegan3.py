from random import choice
from string import ascii_uppercase
import jittor.transform as transform
import os
from configs import global_config, paths_config

from training_stylegan3.coaches.multi_id_coach import MultiIDCoach
from training_stylegan3.coaches.single_id_coach import SingleIDCoach
from utils.ImagesDataset import ImagesDataset


def run_PTI(run_name='', use_wandb=False, use_multi_id_training=False):
    if run_name == '':
        global_config.run_name = ''.join(choice(ascii_uppercase) for i in range(12))
    else:
        global_config.run_name = run_name
    
    global_config.pivotal_training_steps = 1
    global_config.training_step = 1
    
    img_size = 1024
    transform_image = transform.Compose([
        transform.Resize(size = img_size),
        transform.ImageNormalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    align_frames_path = os.path.join(paths_config.input_video_path, 'align_frames')
    dataset = ImagesDataset(align_frames_path, transform_image).set_attrs(batch_size=1, shuffle=False)

    if use_multi_id_training:
        coach = MultiIDCoach(dataset, use_wandb)
    else:
        coach = SingleIDCoach(dataset, use_wandb)

    coach.train()

    return global_config.run_name


if __name__ == '__main__':
    run_PTI(run_name='', use_wandb=False, use_multi_id_training=True)

