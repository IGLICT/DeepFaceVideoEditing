import face_alignment
import jittor as jt
from modules.generator import OcclusionAwareGenerator
from modules.keypoint_detector import KPDetector

import os
import imageio
from animate import normalize_kp
import numpy as np
from tqdm import tqdm
from scipy.spatial import ConvexHull
from argparse import ArgumentParser
from skimage.transform import resize
from skimage import img_as_ubyte
import yaml

jt.flags.use_cuda = 1

def load_checkpoints(config_path, checkpoint_path, cpu=False):
    with open(config_path) as f:
        #config = yaml.load(f)
        config = yaml.safe_load(f)

    generator = OcclusionAwareGenerator(**config['model_params']['generator_params'],
                                        **config['model_params']['common_params'])

    kp_detector = KPDetector(**config['model_params']['kp_detector_params'],
                             **config['model_params']['common_params'])
    
    #checkpoint = jt.load(checkpoint_path)
    checkpoint = jt.load('./weights/jt-vox-adv-cpk.pkl')

    generator.load_state_dict(checkpoint['generator'])
    kp_detector.load_state_dict(checkpoint['kp_detector'])

    generator.eval()
    kp_detector.eval()
    
    return generator, kp_detector

#generator, kp_detector = load_checkpoints(config_path='./config/vox-256.yaml', checkpoint_path="")

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
        #for frame_idx in range(num):
            driving_frame = driving[:, :, frame_idx]
            kp_driving = kp_detector(driving_frame)
            kp_norm = normalize_kp(kp_source=kp_source, kp_driving=kp_driving,
                                   kp_driving_initial=kp_driving_initial, use_relative_movement=relative,
                                   use_relative_jacobian=relative, adapt_movement_scale=adapt_movement_scale)
            out = generator(source, kp_source=kp_source, kp_driving=kp_norm)
            #print(out)
            #out_numpy = out['prediction'].detach().numpy()
            predictions.append(np.transpose(out['prediction'].detach().numpy(), [0, 2, 3, 1])[0])
            #predictions.append(np.transpose(out['deformed'].data.cpu().numpy(), [0, 2, 3, 1])[0])
    return predictions

def find_best_frame(source, driving, cpu=False):
    def normalize_kp(kp):
        kp = kp - kp.mean(axis=0, keepdims=True)
        area = ConvexHull(kp[:, :2]).volume
        area = np.sqrt(area)
        kp[:, :2] = kp[:, :2] / area
        return kp

    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=True,
                                      device='cpu' if cpu else 'cuda')
    kp_source = fa.get_landmarks(255 * source)[0]
    kp_source = normalize_kp(kp_source)
    norm  = float('inf')
    frame_num = 0
    for i, image in tqdm(enumerate(driving)):
        kp_driving = fa.get_landmarks(255 * image)[0]
        kp_driving = normalize_kp(kp_driving)
        new_norm = (np.abs(kp_source - kp_driving) ** 2).sum()
        if new_norm < norm:
            norm = new_norm
            frame_num = i
    return frame_num

#python demo.py --source_image compare_exp/ex4/frame00043.jpg --driving_video compare_exp/ex4/ori.mp4 --result_video compare_exp/ex4/relative.mp4 --relative --find_best_frame
#python demo.py --source_image compare_exp/ex4/frame00043.jpg --driving_video compare_exp/ex4/ori.mp4 --result_video compare_exp/ex4/absolute.mp4 --find_best_frame

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", default='./config/vox-256.yaml', help="path to config")
    parser.add_argument("--checkpoint", default='./weights/jt-vox-adv-cpk.pkl', help="path to checkpoint to restore")

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
    parser.set_defaults(adapt_scale=False)

    opt = parser.parse_args()

    generator, kp_detector = load_checkpoints(config_path='./config/vox-256.yaml', checkpoint_path=opt.checkpoint)
    
    source_image = imageio.imread(opt.source_image)
    reader = imageio.get_reader(opt.driving_video)
    fps = reader.get_meta_data()['fps']
    driving_video = []
    try:
        for im in reader:
            driving_video.append(im)
    except RuntimeError:
        pass
    reader.close()

    source_image = resize(source_image, (256, 256))[..., :3]
    driving_video = [resize(frame, (256, 256))[..., :3] for frame in driving_video]

    if opt.find_best_frame or opt.best_frame is not None:
        i = opt.best_frame if opt.best_frame is not None else find_best_frame(source_image, driving_video, cpu=opt.cpu)
        print ("Best frame: " + str(i))
        driving_forward = driving_video[i:]
        driving_backward = driving_video[:(i+1)][::-1]
        predictions_forward = make_animation(source_image, driving_forward, generator, kp_detector, relative=opt.relative, adapt_movement_scale=opt.adapt_scale)
        predictions_backward = make_animation(source_image, driving_backward, generator, kp_detector, relative=opt.relative, adapt_movement_scale=opt.adapt_scale)
        predictions = predictions_backward[::-1] + predictions_forward[1:]
    else:
        predictions = make_animation(source_image, driving_video, generator, kp_detector, relative=False, adapt_movement_scale=False)

    imageio.mimsave(opt.result_video, [img_as_ubyte(frame) for frame in predictions], fps=fps)
    
