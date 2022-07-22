import cv2
import os

from util.load_mats import load_lm3d
from PIL import Image
import numpy as np
from util.preprocess import align_img

class FAN(object):
    def __init__(self):
        import face_alignment
        self.model = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, device='cuda')
        #self.model.cuda()

    def run(self, image):
        '''
        image: 0-255, uint8, rgb, [h, w, 3]
        return: detected box list
        '''
        out = self.model.get_landmarks(image)
        if out is None:
            return [0], 'kpt68'
        else:
            kpt = out[0].squeeze()
            left = np.min(kpt[:,0]); right = np.max(kpt[:,0]); 
            top = np.min(kpt[:,1]); bottom = np.max(kpt[:,1])
            bbox = [left,top, right, bottom]
            return bbox, 'kpt68'
    
    def run_landmarks(self, image):
        out = self.model.get_landmarks(image)
        if out is None:
            return [0], 'kpt68'
        else:
            kpt = out[0].squeeze()
            return kpt

img = cv2.cvtColor(cv2.imread('./vd034.png'), cv2.COLOR_BGR2RGB)
#result = detector.detect_faces(img)
#keypoints = result[0]['keypoints']
#print(keypoints.shape)

detect = FAN()
#lm = detect.run_landmarks(img)
#print(lm.shape)

def read_data(im_path, lm3d_std, to_tensor=True):
    # to RGB 
    im = Image.open(im_path).convert('RGB')
    W,H = im.size
    img_numpy = cv2.cvtColor(cv2.imread('./vd034.png'), cv2.COLOR_BGR2RGB)
    lm = detect.run_landmarks(img_numpy)
    #lm = np.loadtxt(lm_path).astype(np.float32)
    #lm = lm.reshape([-1, 2])
    lm[:, -1] = H - 1 - lm[:, -1]
    _, im, lm, _ = align_img(im, lm, lm3d_std)
    #if to_tensor:
    #    im = jt.array(np.array(im)/255., dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
    #    lm = jt.array(lm).unsqueeze(0)
    return im, lm

lm3d_std = load_lm3d('./BFM') 
im, lm = read_data('./vd034.png', lm3d_std)
im.save('./align.png')

