#***********************************************************
## DeepFaceVideoEditing hyperparameters

## PTI epoches
max_pti_steps = 5
max_images_to_invert = 1000

## sketch optimization parameters
#w_noise_scale = 0.2
sketch_weight = 20.0
image_weight = 20.0
optimize_steps = 100
use_scp = True

## region fusion layers
start_layer = 3
end_layer = 10

## time window editing parameters
#window size
window_last_time = [30]
#size of transition window
window_change_time = [20]
#edit which frame
window_edit_frame = [80]

## Merge region
merge_hair = False
merge_neck = False

#***********************************************************
## Original E4E and PTI hyperparameters
## Architechture
lpips_type = 'alex'
first_inv_type = 'w+'
optim_type = 'adam'

## Locality regularization
latent_ball_num_of_samples = 1
locality_regularization_interval = 1
use_locality_regularization = False
regulizer_l2_lambda = 0.1
regulizer_lpips_lambda = 0.1
regulizer_alpha = 30
LPIPS_value_threshold = 0.06

## Loss
pt_l2_lambda = 1
pt_lpips_lambda = 1

## Optimization
pti_learning_rate = 3e-4
first_inv_lr = 5e-3
train_batch_size = 1
use_last_w_pivots = False


