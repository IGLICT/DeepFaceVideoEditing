import jittor as jt
import os
import imageio
import numpy as np
from tqdm import tqdm
from scipy.spatial import ConvexHull
from argparse import ArgumentParser
from skimage.transform import resize
from skimage import img_as_ubyte
import yaml

from modules.first_order.modules.generator import OcclusionAwareGenerator
from modules.first_order.modules.keypoint_detector import KPDetector
from modules.first_order.animate import normalize_kp

jt.flags.use_cuda = 1

def load_checkpoints(config_path, checkpoint_path, cpu=False):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    generator = OcclusionAwareGenerator(**config['model_params']['generator_params'],
                                        **config['model_params']['common_params'])

    kp_detector = KPDetector(**config['model_params']['kp_detector_params'],
                             **config['model_params']['common_params'])
    
    checkpoint = jt.load(checkpoint_path)

    generator.load_state_dict(checkpoint['generator'])
    kp_detector.load_state_dict(checkpoint['kp_detector'])

    generator.eval()
    kp_detector.eval()
    
    return generator, kp_detector

def make_animation(source_image, driving_video, generator, kp_detector, relative=True, adapt_movement_scale=True):
    with jt.no_grad():
        predictions = []
        source_numpy = np.array(source_image[np.newaxis].astype(np.float32))
        print(type(source_numpy))
        source = jt.array(source_numpy).permute(0, 3, 1, 2)
        driving = jt.array(np.array(driving_video)[np.newaxis].astype(np.float32)).permute(0, 4, 1, 2, 3)
        kp_source = kp_detector(source)
        kp_driving_initial = kp_detector(driving[:, :, 0])
        
        print(driving.shape[2])
        num = driving.shape[2]
        for frame_idx in tqdm(range(num)):
            driving_frame = driving[:, :, frame_idx]
            kp_driving = kp_detector(driving_frame)
            kp_norm = normalize_kp(kp_source=kp_source, kp_driving=kp_driving,
                                   kp_driving_initial=kp_driving_initial, use_relative_movement=relative,
                                   use_relative_jacobian=relative, adapt_movement_scale=adapt_movement_scale)
            out = generator(source, kp_source=kp_source, kp_driving=kp_norm)
            predictions.append(np.transpose(out['prediction'].detach().numpy(), [0, 2, 3, 1])[0])
    return predictions

def make_mask_animation(source_image, mask_image, driving_video, generator, kp_detector, relative=True, adapt_movement_scale=True, cpu=False):
    with jt.no_grad():
        prediction_masks = []
        source = jt.array(source_image[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
        mask = jt.array(mask_image[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
        driving = jt.array(np.array(driving_video)[np.newaxis].astype(np.float32)).permute(0, 4, 1, 2, 3)
        kp_source = kp_detector(source)
        kp_driving_initial = kp_detector(driving[:, :, 0])

        for frame_idx in tqdm(range(driving.shape[2])):
            driving_frame = driving[:, :, frame_idx]
            kp_driving = kp_detector(driving_frame)
            kp_norm = normalize_kp(kp_source=kp_source, kp_driving=kp_driving,
                                   kp_driving_initial=kp_driving_initial, use_relative_movement=relative,
                                   use_relative_jacobian=relative, adapt_movement_scale=adapt_movement_scale)
            out = generator.warp_mask(source, mask, kp_source=kp_source, kp_driving=kp_norm)
            prediction_masks.append(np.transpose(out['deformed_mask'].detach().numpy(), [0, 2, 3, 1])[0])
    return prediction_masks

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", default='./config/vox-256.yaml', help="path to config")
    parser.add_argument("--checkpoint", default='./modules/first_order/weights/jt-vox-adv-cpk.pkl', help="path to checkpoint to restore")

    parser.add_argument("--source_image", default='sup-mat/source.png', help="path to source image")
    parser.add_argument("--driving_video", default='sup-mat/source.png', help="path to driving video")
    parser.add_argument("--result_video", default='result.mp4', help="path to output")

    parser.add_argument("--relative", dest="relative", action="store_true", help="use relative or absolute keypoint coordinates")
    parser.add_argument("--adapt_scale", dest="adapt_scale", action="store_true", help="adapt movement scale based on convex hull of keypoints")

    parser.add_argument("--find_best_frame", dest="find_best_frame", action="store_true", 
                        help="Generate from the frame that is the most alligned with source. (Only for faces, requires face_aligment lib)")

    parser.add_argument("--best_frame", dest="best_frame", type=int, default=None,  
                        help="Set frame to start from.")
 
    parser.add_argument("--cpu", dest="cpu", action="store_true", help="cpu mode.")


    parser.set_defaults(relative=False)
    parser.set_defaults(adapt_scale=True)

    opt = parser.parse_args()

    from configs import global_config, hyperparameters, paths_config
    video_path = paths_config.input_video_path
    edit_path = os.path.join(video_path, paths_config.inversion_edit_path)

    mask_image = imageio.imread(edit_path + "mask_edit.jpg")
    source_image = imageio.imread(edit_path + "img.jpg")
    video_path = video_path + 'align_frames/'

    result_dir = edit_path + 'mask_warp/'
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    
    driving_video = []
    driving_list = sorted(os.listdir(video_path))
    for im_name in tqdm(range(len(driving_list))):
        image = imageio.imread(video_path + driving_list[im_name])
        driving_video.append(image)

    generator, kp_detector = load_checkpoints(config_path='./modules/first_order/config/vox-256.yaml', checkpoint_path=opt.checkpoint)

    source_image = resize(source_image, (256, 256))[..., :3]
    mask_image = resize(mask_image, (256, 256))[..., :3]
    driving_video = [resize(frame, (256, 256))[..., :3] for frame in driving_video]

    predictions = make_mask_animation(source_image, mask_image, driving_video, generator, kp_detector, relative=False, adapt_movement_scale=False)

    for i in range(len(predictions)):
        frame = predictions[i]
        frame_path = result_dir + "%05d.jpg"%(i)
        imageio.imsave(frame_path, img_as_ubyte(frame))
