import os
import os.path
from skimage.transform import resize
from skimage import img_as_ubyte
from warp_mask import load_checkpoints
from warp_mask import make_mask_animation

import jittor as jt
import jittor.nn as nn
import jittor.transform as transform
from PIL import Image
import numpy as np
import argparse
from tqdm import tqdm
import imageio

from configs import global_config, paths_config, hyperparameters
from models.stylegan3.networks_stylegan3 import Generator
from models.psp_stylegan3 import pSp

from modules.face_recon_jittor.models.networks import define_net_recon
from utils.util import load_old_G, read_img, save_img

def split_coeff(coeffs):
    """
    Return:
        coeffs_dict     -- a dict of torch.tensors

    Parameters:
        coeffs          -- torch.tensor, size (B, 256)
    """
    id_coeffs = coeffs[:, :80]
    exp_coeffs = coeffs[:, 80: 144]
    tex_coeffs = coeffs[:, 144: 224]
    angles = coeffs[:, 224: 227]
    gammas = coeffs[:, 227: 254]
    translations = coeffs[:, 254:]
    return {
        'id': id_coeffs,
        'exp': exp_coeffs,
        'tex': tex_coeffs,
        'angle': angles,
        'gamma': gammas,
        'trans': translations
    }

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
        #new_image = jt.nn.interpolate(image, size=(256, 256), mode='bilinear')
        new_image = image
        buf, w = self.e4e_inversion_net(new_image, randomize_noise=False, return_latents=True, resize=False,
                                      input_code=False)
        del buf
        return w

def get_cos_weight(feature_list, generated_f):
    cos = jt.sum(feature_list * generated_f, dim = 1)
    cos = cos / (jt.sqrt(jt.sum(feature_list * feature_list, dim = 1)) * jt.sqrt(jt.sum(generated_f * generated_f, dim = 1)))
    cos = jt.exp(cos*30)
    cos = cos / jt.sum(cos)
    #print(cos)
    return cos

if __name__ == '__main__':
    video_path = paths_config.input_video_path
    shapePath = os.path.join(video_path, 'edit/baseShape/')
    expPath = os.path.join(video_path, 'edit/exp/')
    windowPath = os.path.join(video_path, 'edit/window/')
    imgPath = os.path.join(video_path, "align_frames/")
    images = sorted(os.listdir(imgPath))

    resultPath = os.path.join(video_path, paths_config.propagation_dir)
    if not os.path.exists(resultPath):
        os.makedirs(resultPath)

    shapePath_list = paths_config.shapePath_list
    expPath_list = paths_config.expPath_list
    windowPath_list = paths_config.windowPath_list
    
    #window size
    window_last_time = hyperparameters.window_last_time
    #size of generating window
    window_change_time = hyperparameters.window_change_time
    #edit which frame
    window_edit_frame = hyperparameters.window_edit_frame

    #-------------------------------  warp mask  ---------------------------------------
    # Warp masks according to the motion and expression
    mask_image_list = []
    source_image_list = []
    mask_warps_list = []
    for shape_edit in shapePath_list:
        source_image_list.append(os.path.join(shapePath, shape_edit, 'img.jpg'))
        mask_image_list.append(os.path.join(shapePath, shape_edit, 'mask_edit.jpg'))
        mask_warps_list.append(os.path.join(shapePath, shape_edit, 'mask_warp'))
    for window_edit in windowPath_list:
        source_image_list.append(os.path.join(windowPath, window_edit, 'img.jpg'))
        mask_image_list.append(os.path.join(windowPath, window_edit, 'mask_edit.jpg'))
        mask_warps_list.append(os.path.join(windowPath, window_edit, 'mask_warp'))
    if len(expPath_list) > 0:
        source_image_list.append(os.path.join(expPath, expPath_list[0], 'img.jpg'))
        mask_image_list.append(os.path.join(expPath, expPath_list[0], 'mask_edit.jpg'))
        mask_warps_list.append(os.path.join(expPath, expPath_list[0], 'mask_warp'))
    
    use_warp = False
    for mask_count in range(len(mask_image_list)):
        if not os.path.exists(mask_warps_list[mask_count]):
            os.makedirs(mask_warps_list[mask_count])
            use_warp = True
    if use_warp:
        # Warp mask
        generator, kp_detector = load_checkpoints(config_path='./modules/first_order/config/vox-256.yaml', 
                                              checkpoint_path='./modules/first_order/weights/jt-vox-adv-cpk.pkl')
        video_path = video_path + 'align_frames/'
        driving_video = []
        driving_list = sorted(os.listdir(video_path))
        for im_name in tqdm(range(len(driving_list))):
            image = imageio.imread(video_path + driving_list[im_name])
            driving_video.append(image)
        driving_video = [resize(frame, (256, 256))[..., :3] for frame in driving_video]
        for mask_count in range(len(mask_image_list)):
            mask_image = imageio.imread(mask_image_list[mask_count])
            mask_image = resize(mask_image, (256, 256))[..., :3]
            source_image = imageio.imread(source_image_list[mask_count])
            source_image = resize(source_image, (256, 256))[..., :3]
            
            print("warp mask: %d/%d"%(mask_count, len(mask_image_list)))
            predictions = make_mask_animation(source_image, mask_image, driving_video, generator, kp_detector, relative=False, adapt_movement_scale=False)
            for i in range(len(predictions)):
                frame = predictions[i]
                frame_path = os.path.join(mask_warps_list[mask_count], "%05d.jpg"%(i))
                imageio.imsave(frame_path, img_as_ubyte(frame))

    #----------------------------- editing propagation ----------------------------- 
    smooth_mask_class = SoftDilate(kernel_size=15, threshold=0.9, iterations=10)
    coach = E4E()
    G_path = os.path.join(paths_config.input_video_path, 'ffhq_weights_stylegan3.pkl')
    stylegan_G = load_old_G(G_path)
    
    print("propagate the editing manipuation")
    ##############################################################################
    #Shape editing: support editing of mutiple operations
    shape_vectors = []
    shape_mask_paths = []
    shape_mask_names = []
    for shape_edit in shapePath_list:
        image = read_img(shapePath + shape_edit + '/img.jpg', 256)
        w_before = coach.get_e4e_inversion(image)

        w_edit = jt.load(shapePath + shape_edit + '/refine_w.pkl')
        w_vector = w_edit - w_before
        shape_vectors.append(w_vector)

        maskPath = shapePath + shape_edit  + "/mask_warp/"
        shape_mask_paths.append(maskPath)
        masks = sorted(os.listdir(maskPath))
        shape_mask_names.append(masks)
    
    ##############################################################################
    #Time Window editing: support editing of mutiple operations
    window_vectors = []
    window_mask_paths = []
    window_mask_names = []
    
    for window_edit in windowPath_list:
        image = read_img(windowPath + window_edit + '/img.jpg', 256)
        w_before = coach.get_e4e_inversion(image)

        w_edit = jt.load(windowPath + window_edit + '/refine_w.pkl')
        w_vector = w_edit - w_before
        window_vectors.append(w_vector)

        maskPath = windowPath + window_edit  + "/mask_warp/"
        window_mask_paths.append(maskPath)
        masks = sorted(os.listdir(maskPath))
        window_mask_names.append(masks)
    
    ##############################################################################
    #Expression guidance editing: support editing of mutiple operations
    if len(expPath_list) > 0:
        net_recon = define_net_recon('resnet50')
        checkpoint = './modules/face_recon_jittor/checkpoints/epoch_20.pkl'
        weights_dict = jt.load(checkpoint)
        net_recon.load_state_dict(weights_dict['net_recon'])
        net_recon.eval()

        expMaskPath = expPath + expPath_list[0] + "/mask_warp/"
        exp_masks = sorted(os.listdir(expMaskPath))

        exp_code_list = jt.zeros([len(expPath_list),64])
        latent_list = []

    for i in range(len(expPath_list)):
        # Get exp code
        im = Image.open(os.path.join(expPath, expPath_list[i], 'img.jpg')).convert('RGB')
        im = im.resize((224,224))
        im = jt.array(np.array(im)/255.).permute(2, 0, 1).unsqueeze(0)
        output_coeff = net_recon(im)
        coeff_dict = split_coeff(output_coeff)
        exp_code_list[i:i+1, :] = coeff_dict['exp']

        # Get the edited latent code
        image = read_img(expPath + expPath_list[i] + '/img.jpg', 256)
        w_before = coach.get_e4e_inversion(image)
        refine_w_path = expPath + expPath_list[i] + '/refine_w.pkl'
        if os.path.exists(refine_w_path):
            w_edit = jt.load(refine_w_path)
        else:
            w_edit = w_before
        w_vector = w_edit - w_before
        latent_list.append(w_vector)
    
    #########################################################################
    # Propagate and fuse all editing operations
    for im_name in tqdm(range(len(images))):
        image = read_img(imgPath + images[im_name], 256)
        w_pivot = coach.get_e4e_inversion(image)

        x_lists = []
        mask_edit_lists = []

        #########################################################################
        #apply shape vector
        for i_shape in range(len(shapePath_list)):
            w_edit = w_pivot + shape_vectors[i_shape]
            #w_edit[:, 10:17, :] = w_pivot_first_frame[:, 10:17, :]
            x_list, img = stylegan_G.synthesis.execute_lists(w_edit.detach(), noise_mode='const')
            x_lists.append(x_list)
            mask_edit = read_img(shape_mask_paths[i_shape] + shape_mask_names[i_shape][im_name], 512)
            mask_edit = mask_edit[:, 0:1,:,:]
            mask_edit = (mask_edit + 1.0) / 2.0
            mask_edit, _ = smooth_mask_class(mask_edit)
            mask_edit_lists.append(mask_edit)
        
        #########################################################################
        #apply time window vector
        for i_window in range(len(windowPath_list)):
            edit_frame = window_edit_frame[i_window]
            lasting_frame = window_last_time[i_window]
            window_frame = window_change_time[i_window]
            w_vector = window_vectors[i_window]
            if im_name >= (edit_frame - lasting_frame - window_frame) and im_name <= (edit_frame + lasting_frame + window_frame):
                if im_name > edit_frame - lasting_frame and im_name < edit_frame + lasting_frame:
                    print("lasting window")
                    w_edit = w_vector
                elif im_name <= edit_frame - lasting_frame:
                    print("generating window")
                    rate = 1.0 - float(abs(im_name - (edit_frame - lasting_frame))/window_frame)
                    w_edit = w_vector * rate
                else:
                    print("ending window")
                    rate = 1.0 - float(abs((edit_frame + lasting_frame) - im_name)/window_frame)
                    w_edit = w_vector * rate
  
                w_edit = w_pivot + w_edit
                #w_edit[:, 10:17, :] = w_pivot_first_frame[:, 10:17, :]
                 
                x_list, img = stylegan_G.synthesis.execute_lists(w_edit , noise_mode='const')
                x_lists.append(x_list)
                mask_edit = read_img(window_mask_paths[i_window] + window_mask_names[i_window][im_name], 512)
                mask_edit = mask_edit[:, 0:1,:,:]
                mask_edit = (mask_edit + 1.0) / 2.0
                mask_edit, _ = smooth_mask_class(mask_edit)
                mask_edit_lists.append(mask_edit)
        
        #########################################################################
        #apply expression vector
        if len(expPath_list) > 0:
            im = Image.open(imgPath + images[im_name]).convert('RGB')
            im = im.resize((224,224))
            im = jt.array(np.array(im)/255.).permute(2, 0, 1).unsqueeze(0)
            output_coeff = net_recon(im)
            coeff_dict = split_coeff(output_coeff)
            exp_code = coeff_dict['exp']
            cos_similarity = get_cos_weight(exp_code_list, exp_code)

            w_vector = jt.zeros(latent_list[0].shape)
            for j in range(0, len(expPath_list)):
                w_vector += cos_similarity[j] * latent_list[j]

            w_edit = w_pivot + w_vector
            #w_edit[:, 10:17, :] = w_pivot_first_frame[:, 10:17, :]
            x_list, img = stylegan_G.synthesis.execute_lists(w_edit, noise_mode='const')
            x_lists.append(x_list)
            mask_edit = read_img(expMaskPath + exp_masks[im_name], 512)
            mask_edit = mask_edit[:, 0:1,:,:]
            mask_edit = (mask_edit + 1.0) / 2.0
            mask_edit, _ = smooth_mask_class(mask_edit)
            mask_edit_lists.append(mask_edit)

        ###########[0   1   2   3   4   5   6    7    8    9    10   11    12    13    #14   #15 ]
        res_list = [16, 16, 16, 32, 32, 64, 128, 128, 256, 256, 512, 1024, 1024, 1024, 1024, 1024]
        start_layer = hyperparameters.start_layer
        end_layer = hyperparameters.end_layer
        generated_images = stylegan_G.synthesis.execute_mask(w_pivot, mask_edit_lists, x_lists, start_list=start_layer, end_list=end_layer, noise_mode='const')

        save_path = os.path.join(resultPath, images[im_name][:-4]+'.jpg')
        save_img(generated_images, save_path)

        #jt.sync_all()
        #jt.display_memory_info()
    #print("running time:", running_time / len(images))


