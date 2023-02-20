import dlib
import PIL
import PIL.Image
import numpy as np
import os
import threading
import time
import cv2

from configs import paths_config, hyperparameters
from scipy import ndimage

def FrameCapture(path, save_dir, start_n=0): 
    # Path to video file 
    vidObj = cv2.VideoCapture(path)
    print("extract_video frames:", path)
    # Used as counter variable 
    count = start_n
    
    # vidObj object calls read 
    # function extract frames 
    success, image = vidObj.read() 
    while success:
        cv2.imwrite(save_dir + "/frame%05d.jpg" % count, image)
        print("\r %d frame" % count, end="")
        count += 1
        success, image = vidObj.read()
    print("")

name_lists = []
lm_List = []
video_dir = paths_config.input_video_path
ori_dir = video_dir + 'img'
align_dir = video_dir + 'align_frames'

# create editing manipulation folder, include 3 class editing
if not os.path.exists(os.path.join(video_dir,'edit/baseShape/')):
    os.makedirs(os.path.join(video_dir,'edit/baseShape/'))
if not os.path.exists(os.path.join(video_dir,'edit/exp/')):
    os.makedirs(os.path.join(video_dir,'edit/exp/'))
if not os.path.exists(os.path.join(video_dir,'edit/window/')):
    os.makedirs(os.path.join(video_dir,'edit/window/'))

# extract frames 
if not os.path.exists(ori_dir):
    os.mkdir(ori_dir)
video_path = os.path.join(video_dir, paths_config.video_name)
FrameCapture(video_path, ori_dir)

image_lists = sorted(os.listdir(ori_dir))
trans_params_dict = {}
trans_params_dict['lm'] = []
trans_params_dict['e2m'] = []
trans_params_dict['x'] = []
trans_params_dict['e2e'] = []
trans_params_dict['quad'] = []
trans_params_dict['midw'] = []
trans_params_dict['midh'] = []
trans_params_dict['crop'] = []
trans_params_dict['pad'] = []

class LandmarksDetector:
    def __init__(self, predictor_model_path):
        """
        :param predictor_model_path: path to shape_predictor_68_face_landmarks.dat file
        """
        self.detector = dlib.get_frontal_face_detector() # cnn_face_detection_model_v1 also can be used
        self.shape_predictor = dlib.shape_predictor(predictor_model_path)

    def get_landmarks(self, image):
        img = dlib.load_rgb_image(image)
        dets = self.detector(img, 1)

        for detection in dets:
            face_landmarks = [(item.x, item.y) for item in self.shape_predictor(img, detection).parts()]
            yield face_landmarks
    
    def run_landmarks(self, image):
        img = image
        dets = self.detector(img, 1)
        
        out = []
        for detection in dets:
            face_landmarks = [(item.x, item.y) for item in self.shape_predictor(img, detection).parts()]
            out.append(face_landmarks)
        return out

class ParserThread(threading.Thread):
    def __init__(self, name, lock):
        super(ParserThread, self).__init__()
        self.name = name
        self.lock = lock
        self.detect = LandmarksDetector(paths_config.dlib_weights)

    def run(self):
        print('%s----Threading begin' % self.name)
        global name_lists

        #while len(image_lists) > 0:
        while True:
            self.lock.acquire()
            if not len(image_lists) > 0:
                self.lock.release()
                break
            #self.lock.acquire()
            img_name = image_lists.pop(0)
            print('%s----detecting…… remain %d items……' % (str(self.name), len(image_lists)))
            self.lock.release()
            self.parse_content(img_name)
        print('%s----Threading end' % self.name)
 
    def parse_content(self, img_name):
        img_numpy = dlib.load_rgb_image(os.path.join(ori_dir, img_name))
        out = self.detect.run_landmarks(img_numpy)
        kpt = np.array(out[0]).squeeze()
        self.lock.acquire()
        name_lists.append(img_name)
        lm_List.append(kpt)
        #lm_List[img_name] = kpt
        self.lock.release()

class MovAvg(object):
    def __init__(self, window_size=7):
        self.window_size = window_size
        self.data_queue = []

    def update(self, data):
        if len(self.data_queue) == self.window_size:
            del self.data_queue[0]
        self.data_queue.append(data)
        return sum(self.data_queue)/len(self.data_queue)

def smooth_sth(sth_dict, window_size, key, savedir=None):
    use_ma = True
    window_size = window_size
    ma  = MovAvg(window_size=window_size)
    file_num = len(sth_dict[key])
    sth_dict['smooth_'+key]= []

    point_idx = 1
    for i in range(file_num):
        lmsm1 = ma.update(sth_dict[key][i])
        sth_dict['smooth_'+key].append(lmsm1)

    ####### align the middle part, delete the first ones and then insert the two sides with the original ones
    for i in range(window_size-1):
        sth_dict['smooth_'+key].pop(0)
        
    for i in range(window_size//2):
        sth_dict['smooth_'+key].insert(i, sth_dict[key][i])
        sth_dict['smooth_'+key].append(sth_dict[key][file_num-window_size//2+i])

    return sth_dict

def dataprep():
    global trans_params_dict
    trans_params_dict = smooth_sth(trans_params_dict, window_size=4, key='lm', savedir=None)

    ## calculate the eye-to-mouth vector 
    for i in range(len(image_lists)):
        lm = trans_params_dict['smooth_lm'][i]
        lm_eye_left = lm[36: 42]  # left-clockwise
        lm_eye_right = lm[42: 48]  # left-clockwise
        lm_mouth_outer = lm[48: 60]  # left-clockwise
        lm_mouth_inner = lm[60: 68]

        # Calculate auxiliary vectors.
        eye_left = np.mean(lm_eye_left, axis=0)
        eye_right = np.mean(lm_eye_right, axis=0)
        # eye_left = lm_eye_left
        # eye_right = lm_eye_right
        eye_avg = (eye_left + eye_right) * 0.5 
        eye_to_eye = eye_right - eye_left

        mouth_left = lm_mouth_outer[0]
        mouth_right = lm_mouth_outer[6]
        # mouth_left = np.mean(lm_mouth_outer, axis=0)
        # mouth_right = np.mean(lm_mouth_inner, axis=0)
        mouth_avg = (mouth_left + mouth_right) * 0.5
        eye_to_mouth = mouth_avg - eye_avg
        
        ###---------------
        trans_params_dict['e2e'].append(eye_to_eye)
        trans_params_dict['e2m'].append(eye_to_mouth)

        ###---------------
        x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
        trans_params_dict['x'].append(x)

    ## smooth
    trans_params_dict = smooth_sth(trans_params_dict, window_size=4, key='x', savedir=None)
    trans_params_dict = smooth_sth(trans_params_dict, window_size=4, key='e2e', savedir=None)
    trans_params_dict = smooth_sth(trans_params_dict, window_size=4, key='e2m', savedir=None)

def align_face(img, lm_load, smooth_e2e, smooth_e2m, smooth_x):
    t_1 = time.time()

    lm = lm_load

    lm_chin = lm[0: 17]  # left-right
    lm_eyebrow_left = lm[17: 22]  # left-right
    lm_eyebrow_right = lm[22: 27]  # left-right
    lm_nose = lm[27: 31]  # top-down
    lm_nostrils = lm[31: 36]  # top-down
    lm_eye_left = lm[36: 42]  # left-clockwise
    lm_eye_right = lm[42: 48]  # left-clockwise
    lm_mouth_outer = lm[48: 60]  # left-clockwise
    lm_mouth_inner = lm[60: 68]  # left-clockwise

    # Calculate auxiliary vectors.
    eye_left = np.mean(lm_eye_left, axis=0)
    eye_right = np.mean(lm_eye_right, axis=0)
    # eye_left = lm_eye_left
    # eye_right = lm_eye_right
    eye_avg = (eye_left + eye_right) * 0.5 
    eye_to_eye = eye_right - eye_left

    mouth_left = lm_mouth_outer[0]
    mouth_right = lm_mouth_outer[6]
    # mouth_left = np.mean(lm_mouth_outer, axis=0)
    # mouth_right = np.mean(lm_mouth_inner, axis=0)
    mouth_avg = (mouth_left + mouth_right) * 0.5
    eye_to_mouth = mouth_avg - eye_avg

    x = smooth_x
    eye_to_eye = smooth_e2e
    eye_to_mouth = smooth_e2m

    #print("debug:x,eye_to_eye,eye_to_mouth",x,eye_to_eye,eye_to_mouth)

    x /= np.hypot(*x)
    x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
    y = np.flipud(x) * [-1, 1]
    c = eye_avg + eye_to_mouth * 0.1
    # print("debug:x,y,c",x,y,c)
    quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
    qsize = np.hypot(*x) * 2

    # print("debug:x,quad,qsize",x,quad,qsize)

    output_size = 1024
    transform_size = 1024
    enable_padding = True

    # Shrink.
    t_2 = time.time()
    shrink = int(np.floor(qsize / output_size * 0.5))
    if shrink > 1:
        print(" ffhq_shrink:",shrink)
        rsize = (int(np.rint(float(img.size[0]) / shrink)), int(np.rint(float(img.size[1]) / shrink)))
        img = img.resize(rsize, PIL.Image.ANTIALIAS)
        print(" ---------------------------ffhq_rsize:",rsize)
        quad /= shrink
        qsize /= shrink
        assert(0)
    else:
        print(" ---------------------------no_ffhq_rsize")

    # Crop.
    t_3 = time.time()
    border = max(int(np.rint(qsize * 0.1)), 3)
    crop = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))), int(np.ceil(max(quad[:, 0]))),
            int(np.ceil(max(quad[:, 1]))))
    crop = (max(crop[0] - border, 0), max(crop[1] - border, 0), min(crop[2] + border, img.size[0]),
            min(crop[3] + border, img.size[1]))
    if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
        print(" ---------------------------ffhq_crop:",crop)
        img = img.crop(crop)
        quad -= crop[0:2]
    else:
        crop = [0,0,img.size[0],img.size[1]]
        print(" ---------------------------no_ffhq_crop")

    # Pad.  Very Slow!!!!!!!
    t_4 = time.time()
    pad = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))), int(np.ceil(max(quad[:, 0]))),
        int(np.ceil(max(quad[:, 1]))))
    pad = (max(-pad[0] + border, 0), max(-pad[1] + border, 0), max(pad[2] - img.size[0] + border, 0),
        max(pad[3] - img.size[1] + border, 0))
    t_4_1 = time.time()
    if enable_padding and max(pad) > border - 4:
        pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
        print(" ---------------------------ffhq_pad:",pad)
        img = np.pad(np.float32(img), ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), 'reflect')
        h, w, _ = img.shape
        y, x, _ = np.ogrid[:h, :w, :1]
        mask = np.maximum(1.0 - np.minimum(np.float32(x) / pad[0], np.float32(w - 1 - x) / pad[2]),
                        1.0 - np.minimum(np.float32(y) / pad[1], np.float32(h - 1 - y) / pad[3]))
        blur = qsize * 0.02
        t_4_2 = time.time()
        img += (ndimage.gaussian_filter(img, [blur, blur, 0]) - img) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
        t_4_3 = time.time()
        img += (np.median(img, axis=(0, 1)) - img) * np.clip(mask, 0.0, 1.0)
        t_4_4 = time.time()
        img = PIL.Image.fromarray(np.uint8(np.clip(np.rint(img), 0, 255)), 'RGB')
        t_4_5 = time.time()
        quad += pad[:2]
        t_4_6 = time.time()

        timelist = [t_4, t_4_1, t_4_2, t_4_3, t_4_4, t_4_5, t_4_6]
        for tt in range(1,len(timelist)):
            print("---- ---- pad time cost %d:"%tt, timelist[tt]-timelist[tt-1])
    else:
        pad = [0,0,0,0]
        print(" ---------------------------no_ffhq_pad")

    # Transform.
    t_5 = time.time()
    width, height = img.size
    # img.save("./debug.png")
    quad_clockwise = quad+0.5
    img = img.transform((transform_size, transform_size), PIL.Image.QUAD, (quad + 0.5).flatten(), PIL.Image.BILINEAR)
    if output_size < transform_size:
        img = img.resize((output_size, output_size), PIL.Image.ANTIALIAS)

    # inverse
    trans_params = {}
    quad_clockwise[1][0], quad_clockwise[3][0] = quad_clockwise[3][0], quad_clockwise[1][0]
    quad_clockwise[1][1], quad_clockwise[3][1] = quad_clockwise[3][1], quad_clockwise[1][1]
    trans_params['quad'] = quad_clockwise
    trans_params['midw'] = width
    trans_params['midh'] = height
    trans_params['lm']   = lm
    trans_params['crop'] = np.array(crop)
    trans_params['pad']  = np.array(pad)

    t_6 = time.time()

    timelist = [t_1, t_2, t_3, t_4, t_5, t_6]
    return img, trans_params

class AlignThread(threading.Thread):
    def __init__(self, name, lock):
        super(AlignThread, self).__init__()
        self.name = name
        self.lock = lock
 
    def run(self):
        print('%s----Threading begin' % self.name)
        global name_lists

        while True:
            self.lock.acquire()
            if not len(image_lists) > 0:
                self.lock.release()
                break
            img_name = image_lists.pop(0)
            pilimg = PIL.Image.open(os.path.join(ori_dir, img_name)).convert('RGB')

            ldmk_x = trans_params_dict['smooth_lm'].pop(0)
            smooth_e2e = trans_params_dict['smooth_e2e'].pop(0)
            smooth_e2m = trans_params_dict['smooth_e2m'].pop(0)
            smooth_x = trans_params_dict['smooth_x'].pop(0)

            print('%s----align face…… remain %d items……' % (str(self.name), len(image_lists)))
            self.lock.release()

            aligned_pilimg, trans_params = align_face(pilimg, ldmk_x, smooth_e2e, smooth_e2m, smooth_x)
            
            self.lock.acquire()
            name_lists.append(img_name)

            trans_params_dict['quad'].append(trans_params['quad'])
            trans_params_dict['midw'].append(trans_params['midw'])
            trans_params_dict['midh'].append(trans_params['midh'])
            trans_params_dict['crop'].append(trans_params['crop'])
            trans_params_dict['pad'].append(trans_params['pad'])
            self.lock.release()

            aligned_pilimg.save(os.path.join(align_dir, img_name))

        print('%s----Threading end' % self.name)

if __name__ == '__main__':
    lock = threading.Lock()
    g_parse_list = []

    for i in range(16):
        tparse = ParserThread('Thread-%d'% i, lock)
        g_parse_list.append(tparse)

    for tparse in g_parse_list:
        tparse.start()

    for tparse in g_parse_list:
        tparse.join()
    
    #global name_lists
    trans_params_dict['lm'] = [lm for _,lm in sorted(zip(name_lists, lm_List))]
    name_lists = []
    image_lists = sorted(os.listdir(ori_dir))
    
    dataprep()
    
    if not os.path.exists(align_dir):
        os.mkdir(align_dir) 
    
    g_align_list = []
    for i in range(16):
        tparse = AlignThread('Thread-%d'% i, lock)
        g_align_list.append(tparse)

    for tparse in g_align_list:
        tparse.start()

    for tparse in g_align_list:
        tparse.join()

    trans_params_dict['lm'] = [lm for _,lm in sorted(zip(name_lists, lm_List))]
    trans_params_dict['quad'] = [lm for _,lm in sorted(zip(name_lists, trans_params_dict['quad']))]
    trans_params_dict['midw'] = [lm for _,lm in sorted(zip(name_lists, trans_params_dict['midw']))]
    trans_params_dict['midh'] = [lm for _,lm in sorted(zip(name_lists, trans_params_dict['midh']))]
    trans_params_dict['crop'] = [lm for _,lm in sorted(zip(name_lists, trans_params_dict['crop']))]
    trans_params_dict['pad'] = [lm for _,lm in sorted(zip(name_lists, trans_params_dict['pad']))]
    np.save(os.path.join(video_dir, "trans_params_dict_new_smooth.npy"), trans_params_dict)



