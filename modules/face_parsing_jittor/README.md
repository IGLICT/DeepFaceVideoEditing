# Jittor version: Face parsing 

**BiSeNet: Bilateral Segmentation Network for Real-time Semantic Segmentation**  
Changqian Yu, Jingbo Wang, Chao Peng, Changxin Gao, Gang Yu, Nong Sang. In [ECCV](https://arxiv.org/abs/1808.00897), 2018.

<p align="center">
<img src="img/teaser.png" width="800px"/>
</p>

## Notes
We implement the semantic segmentation and make-up for facial images. Download the weights and put them in `./checkpoints/`

The Google Drive link: https://drive.google.com/drive/folders/11Z5hTWku3ARkTJOmtnoWpKMqTwKF7V1i?usp=sharing

## Quick start

Generate segmentation results for facial images : <br>
```
python test.py --input_dir ./img/input/ --output_dir ./img/parse/
```

<p align="center">
<img src="img/input/458.jpg" width="300px"/>
<img src="img/parse/458.jpg" width="300px"/>
</p>


Make up for facial images: <br>

```
python make_up.py --input_image ./img/input/458.jpg --output_image ./img/makeup/458.jpg
```
<p align="center">
<img src="img/input/458.jpg" width="300px"/>
<img src="img/makeup/458.jpg" width="300px"/>
</p>

This example change the hair color. Other parts,such as lips can also be changed. 

## Acknowledgements

This repository borrows partially from the [original codes](https://github.com/CoinCheung/BiSeNet) and [face parasing torch](https://github.com/zllrunning/face-parsing.PyTorch) repository.

