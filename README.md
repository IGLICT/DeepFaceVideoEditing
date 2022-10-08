# DeepFaceVideoEditing: Sketch-based Deep Editing of Face Videos<br><sub>Official implementation</sub>

![Teaser image](./img/teaser.jpg)

## Abstract
Sketches, which are simple and concise, have been used in recent deep image synthesis methods to allow intuitive generation and editing of facial images. However, it is nontrivial to extend such methods to video editing due to various challenges, ranging from appropriate manipulation propagation and fusion of multiple editing operations to ensure temporal coherence and visual quality. To address these issues, we propose a novel sketch-based facial video editing framework, in which we represent editing manipulations in latent space and propose specific propagation and fusion modules to generate high-quality video editing results based on StyleGAN3. Specifically, we first design an optimization approach to represent sketch editing manipulations by editing vectors, which are propagated to the whole video sequence using a proper strategy to cope with different editing needs. Specifically, input editing operations are classified into two categories: temporally consistent editing and temporally variant editing. The former (e.g., change of face shape) is applied to the whole video sequence directly, while the latter (e.g., change of facial expression or dynamics) is propagated with the guidance of expression or only affects adjacent frames in a given time window. Since users often perform different editing operations in multiple frames, we further present a region-aware fusion approach to fuse diverse editing effects. Our method supports video editing on facial structure and expression movement by sketch, which cannot be achieved by previous works. Both qualitative and quantitative evaluations show the superior editing ability of our system to existing and alternative solutions.

## Prerequisites

1. System

　- Ubuntu 16.04 or later

　- NVIDIA GPU RTX 3090 + CUDA 11.1 + cudnn 8.0.4

2. Software

　- Python 3.8

　- Jittor. More details in <a href="https://github.com/Jittor/Jittor" target="_blank">Jittor</a>

　- Packages. Note: **cupy-cuda111** is for CUDA 11.1. 

  ```
  pip install -r requirements.txt
  ```
  
  - (Optional) If get a 'cutt' error, please disable 'cutt'. 
  ```
  export use_cutt=0
  ``` 

## Download the pre-trained modules
- Download the lpips models from <a href="https://github.com/ty625911724/Jittor_Perceptual-Similarity-Metric
" target="_blank">[LPIPS-jittor]</a>. 
Put the weights into `./lpips/weights/`
- Download the face parsing models from <a href="https://github.com/ty625911724/jittor-face-parsing
" target="_blank">[Face-parsing-jittor]</a>. 
Put the weights into `./modules/face_parsing_jittor/checkpoints/`
- Download the first-order models from <a href="https://github.com/ty625911724/Jittor-first-order
" target="_blank">[First-order-jittor]</a>. 
Put the weights into `./modules/first_order/weights/`
- Download the dlib alignment models from <a href="http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
" target="_blank">[dlib]</a>. 
Unzip and put the weights into `./weights/`
- Download the face 3D recon models from <a href="https://github.com/ty625911724/Jittor-Deep3DFaceRecon
" target="_blank">[Face-recon-jittor]</a>.
Put the weights into `./modules/face_recon_jittor/checkpoints/`. [Basel Face Model 2009 (BFM09)](https://faces.dmi.unibas.ch/bfm/main.php?nav=1-0&id=basel_face_model) and [the Expression Basis (Exp_Pca.bin)](https://drive.google.com/file/d/1bw5Xf8C12pWmcMhNEu6PtsYVZkVucEN6/view?usp=sharing) should also be downloaded and put into `./modules/face_recon_jittor/BFM/`, organized as in <a href="https://github.com/ty625911724/Jittor-Deep3DFaceRecon
" target="_blank">[Face-recon-jittor]</a>
- Download the StyleGAN3 models <a href="https://github.com/ty625911724/Jittor_StyleGAN3
" target="_blank">[StyleGAN3-jittor]</a>, E4E models for StyleGAN3 <a href="https://github.com/ty625911724/Jittor_E4E
" target="_blank">[E4E-jittor]</a> and DeepFaceVideoEditing weights <a href="https://drive.google.com/drive/folders/15g31av5zR3H0BBaaksmrHhTDGhhwD4gS?usp=sharing
" target="_blank">[Google Drive]</a>. Put the weights into `./weights/`

## Video Editing
Download examples from </a><a href="https://drive.google.com/drive/folders/1MzSDIeu_QnirTNqrG7JyUKxzz0n9Rtj7?usp=sharing
" target="_blank">[Google Drive]</a>. Unzip it and put the video directories in `./video_editings/`. 

For each video example, the original video and editing operations are organized as the following structure:

```
video_editings
│
└─── example1
    └─── XXX.mp4
    └─── edit
         └─── baseShape
         |    └─── edit1
         |    │    └─── img.jpg
         |    │    └─── sketch_edit.jpg
         |    │    └─── mask_edit.jpg
         |    └─── edit2
         |    └─── ...
         └─── window
         └─── exp
    |
    └─── ...
```

**Modify the `./configs/paths_config.py`**: change the `input_video_path` to video example directory and `video_name` to the name of input video. (Default settings are for example1)

### Preprocess videos
Extract and align all frames from input video. 

  ```
  python video_align.py
  ```

The generated aligned frames will be in `align_frames` directory for each example. 

### PTI training
In order to recontruct input video, use PTI method to finetune StyleGAN3 generator. 

  ```
  python run_pti_stylegan3.py
  ```

PTI weights will be generated in example directory and pti results for 1st frame will be generated in `pti_results` directory. 

### Sketch editing

Generate the sketch editing results for single frame. 

**Modify the `./configs/paths_config.py`**: change the `inversion_edit_path` to sketch editing directory which contains image, sketch and mask. 

The edit frame is named `img.jpg`, drawn sketch is named `sketch_edit.jpg` and drawn mask is named `mask_edit.jpg`. 

Then, generate sketch optimization results: 

  ```
  python run_sketch.py
  ```

The edited results will be generated in `inversion_edit_path` directory, including `initial_w.pkl`, `refine_w.pkl`, `refine.jpg` and `mask_fusion_result.jpg`. 

Note:

-- The sketch weights and RGB weights could be tuned in `./configs/hyperparameters.py` to generate the better results. 

-- And for each editing operations, this script should be run again with different `./configs/paths_config.py`. For example, this script should be run 3 times if 3 editing operations are applied for a single video. Please read the `readme.txt` for each example. 

### Editing propagation
Before propagating the editing effect, the editing vectors should be generated using the above approach. 

**Modify the `./configs/paths_config.py`**, corresponding to 3 directories in `./video_editings/exampleXXX/edit/`.

- BaseShape editing: Set operation directories in `shapePath_list`
- Time window editing: Set operation directories in `windowPath_list`
- Expression Guidance editing: Set operation directories in `expPath_list`

The time window parameters are set in  `./configs/hyperparameters.py`. 

Then, generate the propagation results: 

  ```
  python run_editing.py
  ```

Edited frames will be saved in `edit/edit_video` directory.

### Face Merging and realignment
Merge the face regions and realign generated frames into original frames. 

  ```
  python video_merge.py
  ```

Merged frames will be saved in `merge_images` directory. Final edited videos will be generated named as `merged.mp4`. 

## Citation

If you found this code useful please cite our work as:

    @article {DeepFaceVideoEditing2022,
    author = {Liu, Feng-Lin and Chen, Shu-Yu and Lai, Yu-Kun and Li, Chunpeng and Jiang, Yue-Ren and Fu, Hongbo and Gao, Lin},
    title = {{DeepFaceVideoEditing}: Sketch-based Deep Editing of Face Videos},
    journal = {ACM Transactions on Graphics (Proceedings of ACM SIGGRAPH 2022)},
    year = {2022},
    volume = 41,
    pages = {167:1--167:16},
    number = 4
    }


