import os
import jittor as jt
import jittor.nn as nn
import jittor.transform as transform
import PIL
import PIL.Image
from modules.face_parsing_jittor.model import BiSeNet
import numpy as np
import cv2
import threading
import time
from tqdm import tqdm
from argparse import ArgumentParser
from configs import global_config, paths_config, hyperparameters

jt.flags.use_cuda = 1

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
        
        #torch.cuda.empty_cache()
        return x, y

class MaskMaster(nn.Module):
    def __init__(self, args):
        super(MaskMaster, self).__init__()
        ## face parsing net
        n_classes = 19
        self.bisenet = BiSeNet(n_classes=n_classes)
        save_pth = os.path.join('./modules/face_parsing_jittor/checkpoints', '79999_iter.pkl')
        self.bisenet.load_state_dict(jt.load(save_pth))
        self.bisenet.eval()

        self.transform_image = transform.Compose([
        transform.ImageNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.merge_neck = args.merge_neck
        self.merge_hair = args.merge_hair

    def encode_segmentation_rgb(self, segmentation, no_neck=True):
        parse = segmentation
        ###     1        2         3        4        5        6        7        8        9       10      11       12       13       14       15       16      17      18
        ### ['skin', 'l_brow', 'r_brow', 'l_eye', 'r_eye', 'eye_g', 'l_ear', 'r_ear', 'ear_r', 'nose', 'mouth', 'u_lip', 'l_lip', 'neck', 'neck_l', 'cloth', 'hair', 'hat']
        #face_part_ids = [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 13]
        face_part_ids = [1, 2, 3, 4, 5, 6, 10, 12, 13]
        if self.merge_neck:
            face_part_ids.append(14)
            face_part_ids.append(15)
        if self.merge_hair:
            face_part_ids.append(17)
        
        mouth_id = 11
        face_map = np.zeros([parse.shape[0], parse.shape[1]])
        mouth_map = np.zeros([parse.shape[0], parse.shape[1]])

        for valid_id in face_part_ids:
            valid_index = np.where(parse==valid_id)
            face_map[valid_index] = 255

        valid_index = np.where(parse==mouth_id)
        mouth_map[valid_index] = 255

        # valid_index = np.where(parse==hair_id)
        # hair_map[valid_index] = 255
        #return np.stack([face_map, mouth_map,hair_map], axis=2)
        return np.stack([face_map, mouth_map], axis=2)

    def getfaceparsing(self, imgs_t):
        #####
        # input: [B,3,H,W], [-1,1]
        #####
        #source_img = ((imgs_t + 1) / 2)
        source_img_norm = jt.array(self.transform_image(imgs_t)).unsqueeze(0)

        #source_img_norm = self.spNorm(source_img)
        source_img_512 = nn.interpolate(source_img_norm, size=(512,512))
        out = self.bisenet(source_img_512)[0]
        parsing = out.squeeze(0).detach().numpy().argmax(0)
        vis_parsing_anno = parsing.astype(np.uint8)  # (512, 512)

        tgt_mask = self.encode_segmentation_rgb(vis_parsing_anno)  # (512, 512, 2)
        mask_tensor = tgt_mask.transpose((2, 0, 1)).astype(np.float32) * (1/255.0)
        face_mask_tensor = mask_tensor[0] + mask_tensor[1]  # torch.Size([512, 512])
        
        #face_mask = face_mask_tensor.numpy()
        face_mask = face_mask_tensor
        face_mask = face_mask[:, :, np.newaxis] * 255

        face_mask = face_mask.astype(np.uint8)
        Unisize = 1024
        face_mask = cv2.resize(face_mask,(Unisize, Unisize))[:,:, np.newaxis]
        return face_mask
    
class face_projector(nn.Module):
    def __init__(self):
        super(face_projector, self).__init__()

        self.smooth_mask_class = SoftDilate(kernel_size=35, threshold=0.9, iterations=10)
        #self.smooth_mask_class = SoftDilate(kernel_size=15, threshold=0.9, iterations=5).cuda()
    
    def merge_npmask(self, npmask1, npmask2):
        merged_mask = npmask1.astype(np.int32) + npmask2.astype(np.int32)
        merged_mask = np.clip(merged_mask, 0, 255).astype(np.uint8)
        if merged_mask.shape[2] ==1:
            merged_mask = np.repeat(merged_mask, 3, axis=2) # (1024, 1024, 3)
        return merged_mask 
    
    def smooth_npmask(self, npmask, smooth_mask):
        with jt.no_grad():
            face_mask_tensor = jt.array(npmask.transpose((2, 0, 1))).float() * (1/255.0)
            face_mask_tensor = jt.mean(face_mask_tensor, dim=0, keepdims=True)
            soft_face_mask_tensor, _ = self.smooth_mask_class(face_mask_tensor.unsqueeze(0))
            #print(soft_face_mask_tensor.shape)
            soft_face_mask_tensor = soft_face_mask_tensor.squeeze(0).squeeze(0)  # torch.Size([512, 512])
            #print(soft_face_mask_tensor.shape)
            soft_face_mask = soft_face_mask_tensor.numpy()

        soft_face_mask = soft_face_mask[:, :, np.newaxis] * 255
        return soft_face_mask.astype(np.uint8)

    def find_coeffs(self, pa, pb):
        matrix = []
        for p1, p2 in zip(pa, pb):
            matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0]*p1[0], -p2[0]*p1[1]])
            matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1]*p1[0], -p2[1]*p1[1]])

        A = np.matrix(matrix, dtype=np.float32)
        B = np.array(pb).reshape(8)

        res = np.dot(np.linalg.inv(A.T * A) * A.T, B)
        return np.array(res).reshape(8)

    def realignment(self, pil_img, width, height, coeffs, unpad, top, bottom, left, right):
        '''
            input: pil img
            return: cv2 img
        '''
        ######## untransform ---------------------------------------------------------------------------------------------------
        pil_img = pil_img.transform((width, height), PIL.Image.PERSPECTIVE, coeffs, PIL.Image.BILINEAR)

        ######## unpad
        pil_img = pil_img.crop(unpad)

        ######## uncrop
        img_return = np.array(pil_img)
        img_return = cv2.copyMakeBorder(img_return, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0,0,0) )

        return img_return

    def transformback(self, pil_img, pil_img_ori, quad_clockwise, width, height, crop, pad, mask_dict, ifsavecomplex=False):
        ########   merge mask   ########
        ori_parsing_hard = mask_dict['ori_parsing_hard'] 
        gen_parsing_hard = mask_dict['gen_parsing_hard'] 
        mask_bg_np = mask_dict['mask_bg_np'] 
        # frontalmask_np = mask_dict['frontalmask_np']
        # ori_frontalmask_np = mask_dict['ori_frontalmask_np']

        ###Union two faces
        merged_mask = self.merge_npmask(ori_parsing_hard, gen_parsing_hard)

        k = np.ones((3, 3), np.uint8)
        merged_mask = cv2.erode(merged_mask, k, iterations=7)
        
        start = time.time()
        merged_mask = self.smooth_npmask(merged_mask, self.smooth_mask_class)
        end = time.time()
        #print('smooth: ', end - start)

        start = time.time()
        merged_mask = merged_mask.astype(np.float32) * (mask_bg_np.astype(np.float32)/255.)
        merged_mask = merged_mask.astype(np.uint8)

        ######## pre parameters ########
        pil_img = pil_img.resize((1024,1024))
        img_ori = np.array(pil_img_ori)
        
        coeffs = self.find_coeffs(quad_clockwise, [[0,0], [1024,0], [1024, 1024], [0,1024]])
        unpad = (pad[0], pad[1], width-pad[2], height-pad[3])
        
        ori_height, ori_width, _ = img_ori.shape
        top = crop[1]
        bottom = ori_height - crop[3]
        left = crop[0]
        right = ori_width - crop[2]

        ### align ###
        img_return = self.realignment(pil_img, width, height, coeffs, unpad, top, bottom, left, right)
        end = time.time()
        #print('realign: ', end - start)

        merged_pil = PIL.Image.fromarray(merged_mask).resize((1024,1024))
        #print(width, " ", height)
        final_mask_align_np = self.realignment(merged_pil, width, height, coeffs, unpad, top, bottom, left, right)

        #### add #### 
        a = img_ori.astype(np.int32)*(1.0 - final_mask_align_np.astype(np.float32)/255.) 
        b = img_return.astype(np.int32)*(final_mask_align_np.astype(np.float32)/255.) 
        img_return_merge = a + b 

        return img_return_merge.astype(np.uint8)[..., ::-1], final_mask_align_np

class MergeThread(threading.Thread):
    def __init__(self, name, lock, ori_dir, gen_dir, merge_dir, merge_dir_mask):
        super(MergeThread, self).__init__()
        self.name = name
        self.lock = lock
        self.face_projector = face_projector()
        self.gen_dir = gen_dir
        self.ori_dir = ori_dir
        self.merge_dir = merge_dir
        self.merge_dir_mask = merge_dir_mask
 
    def run(self):
        print('%s----Threading begin' % self.name)
        #global image_lists

        #while len(image_lists) > 0:
        while True:
            self.lock.acquire()
            if not len(image_lists) > 0:
                self.lock.release()
                break
            img_name = image_lists.pop(0)
            gen_parsing_hard = gen_parsing_list.pop(0)
            ori_parsing_hard = ori_parsing_list.pop(0)
            quad_clockwise = trans_params_dict_load['quad'].pop(0)
            width = trans_params_dict_load['midw'].pop(0)
            height = trans_params_dict_load['midh'].pop(0)
            crop = trans_params_dict_load['crop'].pop(0)
            pad = trans_params_dict_load['pad'].pop(0)
            print('%s----Merging…… remain %d items……' % (str(self.name), len(image_lists)))
            self.lock.release()
            
            blur_edge_length = 30
            shixin = 1024-blur_edge_length*2
            mask_bg = np.full((shixin, shixin, 3), 255)
            mask_bg = cv2.copyMakeBorder(mask_bg, blur_edge_length, blur_edge_length, blur_edge_length, blur_edge_length, cv2.BORDER_CONSTANT, value=(0,0,0) )
            mask_bg_np = cv2.blur(mask_bg, (blur_edge_length*2, blur_edge_length*2)).astype(np.uint8)
            
            ## merge
            mask_dict = {'mask_bg_np':mask_bg_np,
                        'ori_parsing_hard':ori_parsing_hard,
                        'gen_parsing_hard':gen_parsing_hard 
                        }
            
            #end_time = time.time()
            #print("bk running time: ", end_time - start_time)
            
            pilimg_ori = PIL.Image.open(os.path.join(self.ori_dir, img_name)).convert('RGB')
            pilimg_gen = PIL.Image.open(os.path.join(self.gen_dir, img_name)).convert('RGB')
            
            #start_time = time.time()
            cv2_img_add, final_mask_align_np = self.face_projector.transformback(pilimg_gen, pilimg_ori, \
                                            quad_clockwise= quad_clockwise, \
                                            width= width,
                                            height= height,
                                            crop= crop,
                                            pad= pad,
                                            mask_dict=mask_dict,
                                            ifsavecomplex=False)
            
            cv2.imwrite(os.path.join(self.merge_dir, img_name), cv2_img_add)
            cv2.imwrite(os.path.join(self.merge_dir_mask, img_name), final_mask_align_np)

            #self.parse_content(img_name)
        print('%s----Threading end' % self.name)

image_lists = []
gen_parsing_list = []
ori_parsing_list = []
trans_params_dict_load = {}

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--merge_hair", type=bool, default = False, help = "merge hair or not")
    parser.add_argument("--merge_neck", type=bool, default = False, help = "merge neck or not")
    args = parser.parse_args()
    
    #*********************************************************************************************
    ####-----------------transformback------------------------
    video_dir = paths_config.input_video_path
    extract_dir = os.path.join(video_dir, 'img/')
    align_dir = os.path.join(video_dir, 'align_frames/')
    #gen_dir = os.path.join(video_dir, 'edit/edit_video1/')
    gen_dir = os.path.join(video_dir, paths_config.propagation_dir)
    merge_dir = os.path.join(video_dir, paths_config.merge_dir)
    merge_dir_mask = os.path.join(video_dir, 'merge_masks/')
    if not os.path.exists(merge_dir):
        os.mkdir(merge_dir) 
    if not os.path.exists(merge_dir_mask):
        os.mkdir(merge_dir_mask)
    
    img_ori_paths = [os.path.join(extract_dir,name) for name in sorted(os.listdir(extract_dir))]
    img_align_paths = [os.path.join(align_dir,name) for name in sorted(os.listdir(align_dir))]
    img_gen_paths = [os.path.join(gen_dir,name) for name in sorted(os.listdir(gen_dir))]
    image_lists = sorted(os.listdir(extract_dir))
    
    save_dir = video_dir
    trans_params_dict_path = os.path.join(save_dir, "trans_params_dict_new_smooth.npy")
    trans_params_dict_load = np.load(trans_params_dict_path, allow_pickle=True).item()
    
    ####-----------------Gen mask------------------------
    MaskMaster = MaskMaster(args)
    for i in tqdm(range(len(img_ori_paths))):
        img_name = os.path.basename(img_ori_paths[i])
        pilimg_align = PIL.Image.open(img_align_paths[i]).convert('RGB')
        pilimg_gen = PIL.Image.open(img_gen_paths[i]).convert('RGB')
        
        with jt.no_grad():
            gen_parsing_hard = MaskMaster.getfaceparsing(pilimg_align)        
            ori_parsing_hard = MaskMaster.getfaceparsing(pilimg_gen)
        
        gen_parsing_list.append(gen_parsing_hard)
        ori_parsing_list.append(ori_parsing_hard)
    
    ####---------------Merge and realign------------------------
    lock = threading.Lock()
    g_merge_list = []
    for i in range(16):
        tparse = MergeThread('Thread-%d'% i, lock, ori_dir=extract_dir, gen_dir=gen_dir, merge_dir=merge_dir,merge_dir_mask=merge_dir_mask)
        g_merge_list.append(tparse)

    for tparse in g_merge_list:
        tparse.start()

    for tparse in g_merge_list:
        tparse.join()
    
    ### img2vid
    merged_paths = [os.path.join(merge_dir, name) for name in sorted(os.listdir(merge_dir))]
    frame = cv2.imread(merged_paths[0])
    height, width, channels = frame.shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Be sure to use lower case
    out = cv2.VideoWriter(os.path.join(save_dir, paths_config.merge_video_name), fourcc, 25.0, (width, height))
    
    i = 0
    for i in tqdm(range(0,len(merged_paths))):
        frame = cv2.imread(merged_paths[i]) #[:,512:,:]
        out.write(frame)

    # Release everything if job is finished
    out.release()
    
    