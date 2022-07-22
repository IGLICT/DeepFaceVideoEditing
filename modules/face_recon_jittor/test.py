import jittor as jt
from models.networks import define_net_recon
from models.bfm import ParametricFaceModel
import numpy as np
import trimesh
import cv2
import os
from PIL import Image
import numpy as np
from argparse import ArgumentParser

from util.load_mats import load_lm3d
from util.preprocess import align_img

import dlib

jt.flags.use_cuda = 1

class LandmarksDetector:
    def __init__(self, predictor_model_path):
        """
        :param predictor_model_path: path to shape_predictor_68_face_landmarks.dat file
        """
        self.detector = dlib.get_frontal_face_detector() # cnn_face_detection_model_v1 also can be used
        #self.detector = dlib.cnn_face_detection_model_v1('./pretrained_models/mmod_human_face_detector.dat') # cnn_face_detection_model_v1 also can be used
        self.shape_predictor = dlib.shape_predictor(predictor_model_path)

    def get_landmarks(self, image):
        img = dlib.load_rgb_image(image)
        dets = self.detector(img, 1)

        for detection in dets:
            face_landmarks = [(item.x, item.y) for item in self.shape_predictor(img, detection).parts()]
            yield face_landmarks
    
    def run_landmarks(self, image):
        #img = dlib.load_rgb_image(image)
        img = image
        dets = self.detector(img, 1)

        for detection in dets:
            face_landmarks = [(item.x, item.y) for item in self.shape_predictor(img, detection).parts()]
            #face_landmarks = [(item.x, item.y) for item in self.shape_predictor(img, detection.rect).parts()]
            return np.array(face_landmarks)

def read_data_align(im_path, lm, lm3d_std, to_tensor=True):
    # to RGB 
    im = Image.open(im_path).convert('RGB')
    W,H = im.size
    #img_numpy = cv2.cvtColor(cv2.imread('./vd034.png'), cv2.COLOR_BGR2RGB)
    #lm = detect.run_landmarks(img_numpy)
    lm[:, -1] = H - 1 - lm[:, -1]
    _, im, lm, _ = align_img(im, lm, lm3d_std)
    if to_tensor:
        im = jt.array(np.array(im)/255., dtype=jt.float32).permute(2, 0, 1).unsqueeze(0)
        lm = jt.array(lm).unsqueeze(0)
    return im, lm

def read_data(im_path, to_tensor=True):
    # to RGB 
    im = Image.open(im_path).convert('RGB')
    W,H = im.size
    im = im.resize((224,224))
    if to_tensor:
        im = jt.array(np.array(im)/255., dtype=jt.float32).permute(2, 0, 1).unsqueeze(0)
    return im

def save_mesh(pred_vertex, pred_color, name):
    recon_shape = pred_vertex  # get reconstructed shape
    recon_shape[..., -1] = 10 - recon_shape[..., -1] # from camera space to world space
    recon_shape = recon_shape.numpy()[0]
    recon_color = pred_color
    recon_color = recon_color.numpy()[0]
    tri = facemodel.face_buf
    mesh = trimesh.Trimesh(vertices=recon_shape, faces=tri, vertex_colors=np.clip(255. * recon_color, 0, 255).astype(np.uint8))
    mesh.export(name)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--checkpoint", default='./checkpoints/epoch_20.pkl', help="path to checkpoint to restore")
    parser.add_argument("--input_dir", default='./imgs/no_align', help="dir to source images")
    parser.add_argument("--output_dir", default='./imgs/output', help="dir to face parsing images")
    parser.add_argument("--align", action="store_true", help="align images or not")

    opt = parser.parse_args()

    model = define_net_recon('resnet50')
    weights_dict = jt.load(opt.checkpoint)
    model.load_state_dict(weights_dict['net_recon'])
    model.eval()

    facemodel = ParametricFaceModel(is_train=False)
    
    if opt.align:
        lm3d_std = load_lm3d('./BFM')
        detect = LandmarksDetector('./checkpoints/shape_predictor_68_face_landmarks.dat')

    img_list = sorted(os.listdir(opt.input_dir))
    if not os.path.exists(opt.output_dir):
        os.makedirs(opt.output_dir)
    for img_count in range(len(img_list)):
        image_path = os.path.join(opt.input_dir, img_list[img_count])
        result_path = os.path.join(opt.output_dir, img_list[img_count][:-4] + '.obj')
        if opt.align:
            print(image_path)
            img_numpy = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
            lm = detect.run_landmarks(img_numpy)
            img, _ = read_data_align(image_path, lm, lm3d_std)
        else:
            img = read_data(image_path)
        with jt.no_grad():
            output_coeff = model(img)
            pred_vertex, pred_tex, pred_color, pred_lm = facemodel.compute_for_render(output_coeff)
        save_mesh(pred_vertex, pred_color, result_path)
        
        align_path = os.path.join(opt.output_dir, img_list[img_count])
        align_image = np.transpose(img.numpy(), [0, 2, 3, 1]).squeeze(0) * 255.0
        align_image = cv2.cvtColor(align_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(align_path, align_image)

#python test.py --input_dir ./imgs/no_align --output_dir ./imgs/output --align
#python test.py --input_dir ./imgs/align --output_dir ./imgs/output

