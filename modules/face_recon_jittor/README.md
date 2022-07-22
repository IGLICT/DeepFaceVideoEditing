# Jittor version: Deep3DFaceRecon

**Accurate 3D Face Reconstruction with Weakly-Supervised Learning: From Single Image to Image Set**  
  In [CVPRW on AMFG](https://arxiv.org/abs/1903.08527), 2019 (Best Paper Award!).

<p align="center">
<img src="images/example.gif" width="800px"/>
</p>

## Notes
We implement the 3D face reconstruction for facial images. The weights are converted from [original code](https://github.com/sicxu/Deep3DFaceRecon_pytorch). 

## Prepare prerequisite models
1. Our method uses [Basel Face Model 2009 (BFM09)](https://faces.dmi.unibas.ch/bfm/main.php?nav=1-0&id=basel_face_model) to represent 3d faces. Get access to BFM09 using this [link](https://faces.dmi.unibas.ch/bfm/main.php?nav=1-2&id=downloads). After getting the access, download "01_MorphableModel.mat". In addition, we use an Expression Basis provided by [Guo et al.](https://github.com/Juyong/3DFace). Download the Expression Basis (Exp_Pca.bin) using this [link (google drive)](https://drive.google.com/file/d/1bw5Xf8C12pWmcMhNEu6PtsYVZkVucEN6/view?usp=sharing). Organize all files into the following structure:
```
Deep3DFaceRecon_pytorch
│
└─── BFM
    │
    └─── 01_MorphableModel.mat
    │
    └─── Exp_Pca.bin
    |
    └─── ...
```
2. We provide a jittor verison of pretrained model. Download the model using this [link (google drive)](https://drive.google.com/drive/folders/1Kh6MEuOGYMmepsOJo0NA09Xwtb9qqrXd?usp=sharing).
Then, put the weights in `.\checkpoints\` directory.

3. We use dlib to align the facial images. Download the model using this [link (Official website)](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2). 
Then, put the weights in `.\checkpoints\` directory.

## Quick start

Generate the reconstruction `.obj` model of facial images without alignment. The recontruction model is shown by MeshLab.  <br>
```
python test.py --input_dir ./imgs/align --output_dir ./imgs/output
```

<p align="center">
<img src="imgs/align/14877.png" width="200px"/>
<img src="images/no_align.png" width="200px"/>
</p>

Generate the reconstruction `.obj` model of facial images with alignment. The input image is aligned by dlib. The recontruction model is shown by MeshLab. <br>

```
python test.py --input_dir ./imgs/no_align --output_dir ./imgs/output --align
```
<p align="center">
<img src="imgs/no_align/vd034.png" width="200px"/>
<img src="images/vd034.png" width="200px"/>
<img src="images/align.png" width="200px"/>
</p>


## Acknowledgements

This repository borrows partially from the [torch codes](https://github.com/sicxu/Deep3DFaceRecon_pytorch) and [tensorflow codes](https://github.com/microsoft/Deep3DFaceReconstruction) repository.

