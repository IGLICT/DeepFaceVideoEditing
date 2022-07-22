## Pretrained models paths
e4e = './weights/e4e_ffhq_encode_stylegan3.pt'
StyleGAN3_weights = './weights/jt_stylegan3_ffhq_weights_t.pkl'
scp_weights = './weights/jt_scp_stylegan3.pkl'
sketch_branch_weights = './weights/jt_stylegan_sketch.pkl'
dlib_weights = './weights/shape_predictor_68_face_landmarks.dat'

## Keywords
multi_id_model_type = 'multi_id'

## Input Videos name
video_name = 'ori.mp4'
## Input Vides directory
input_video_path = './video_editings/example1/'
## Optimize editing directory
inversion_edit_path = 'edit/baseShape/edit1/'
#inversion_edit_path = 'edit/window/edit1/'

## propagation and fusion operations, list of folders in 
## If no operation is applied, just set the list empty. 
#---------------------BaseShape editing---------------------
shapePath_list = ['edit1']
#------------------Expression guidance editing---------------------
# If has editing operations:
# At least two folders, one with editing operation, another without editing in natural expression
expPath_list = []
#---------------------Time window editing----------------------
windowPath_list = ['edit1']

## Propagation frames directories
propagation_dir = 'edit/edit_video'
## Final Merge frames path and video name
merge_dir = 'merge_images/'
merge_video_name = 'merged.mp4'


