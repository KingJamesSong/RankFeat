# RankFeat
NeurIPS22 "RankFeat: Rank-1 Feature Removal for Out-of-distribution Detection"

## ID/OOD Dataset Preparation

**In-Distribution (ID) dataset.** Please download [ImageNet-1k](http://www.image-net.org/challenges/LSVRC/2012/index) and use its use the validation set as the ID set. 

**Out-of-Distribution (OOD) dataset.** For the used OOD datasets (iNaturalist, SUN, Places, and Textures), please download them from the following links:

```bash
wget http://pages.cs.wisc.edu/~huangrui/imagenet_ood_dataset/iNaturalist.tar.gz
wget http://pages.cs.wisc.edu/~huangrui/imagenet_ood_dataset/SUN.tar.gz
wget http://pages.cs.wisc.edu/~huangrui/imagenet_ood_dataset/Places.tar.gz
wget https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz
```

For the large-scale [Species](https://arxiv.org/pdf/1911.11132.pdf) dataset, please download the four subsets (Protozoa, Microorganisms, Plants, and Mollusks) from the [official link](https://drive.google.com/drive/folders/1j6l7jfGbKL5P5acwKVyktn4y8bWSTeAJ?usp=sharing).

## Pre-trained Model Preparation

For SqueezeNet, it is already available in the Pytorch library. For BiT-S ResNetv2-101 and T2T-ViT-24, one can download the BiT-S ResNetv2-101 and T2T-ViT by the following links:

```bash
wget http://pages.cs.wisc.edu/~huangrui/finetuned_model/BiT-S-R101x1-flat-finetune.pth.tar
wget https://github.com/yitu-opensource/T2T-ViT/releases/download/main/82.6_T2T_ViTt_24.pth.tar
```

For more BiT pre-trained models, one can also refer to [BiT-S pre-trained families](https://github.com/google-research/big_transfer).

## Usage

## Main Results

## Citation

If you think the code is helpful to your research, please consider citing our paper:

```
@inproceedings{song2022rankfeat,
  title={RankFeat: Rank-1 Feature Removal for Out-of-distribution Detection},
  author={Song, Yue and Sebe, Nicu and Wang, Wei},
  booktitle={NeurIPS},
  year={2022}
}
```

The code is built on MOS and GradNorm. If you have any questions or suggestions, please feel free to contact me via `yue.song@unitn.it`.

