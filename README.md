# ACPL-FSL

Code to pretrain, finetune, and evaluate ACPL-FSL on standard few-shot and cross-domain (CDFSL) benchmarks.
Paper: Semi-Supervised Few-Shot Learning via Ensemble-Based Adaptive Clustering-aware Pseudo-Labeling
---

## Dependencies

Python 3.8+ (3.10 recommended)

PyTorch (tested with 1.13.1)

torchvision (tested with 0.14.1)

scikit-learn ≥ 1.0

numpy, scipy, pandas, Pillow

scikit-image, tqdm

opencv-python (cv2)

haven-ai (logging/checkpoint helpers)
(Optional) faiss-cpu/faiss-gpu for faster nearest-neighbor/k-means 

pip install -r requirements.txt


## 1) Setup

```bash
# create env as you like, then:
pip install -r requirements.txt
```


## 2) Datasets

Put datasets on disk and point JSON config *_root fields to your paths.

Example layout (you can use any base path as long as configs match):
data/
├─ mini-imagenet/
├─ tiered-imagenet/
├─ CUB/
├─ cifar-fs/
└─ CDFSL/
   ├─ EuroSAT/
   ├─ ISIC/
   ├─ CropDiseases/
   └─ ChestX/


### Download the Datasets

* [mini-imagenet](https://github.com/renmengye/few-shot-ssl-public#miniimagenet) 
* [tiered-imagenet](https://github.com/renmengye/few-shot-ssl-public#tieredimagenet)
* [CUB](https://github.com/wyharveychen/CloserLookFewShot/tree/master/filelists/CUB)
* [cifar-fs](https://github.com/bertinetto/r2d2#cifar-fs)

Following the [FWT-repo](https://github.com/hytseng0509/CrossDomainFewShot) to download and set up all CDFSL datasets (ChestX, CropDiseases, EuroSAT, ISIC).

## 3) Configs

Configs live in configs/ (and configs/cdfsl/ for cross-domain).
Ensure each config’s dataset_*_root points to your actual folders.


# 4) Usage

## 4.1. Pretrain (single-domain example)
```
python train_runner.py configs/pretrain_miniimagenet_wrn_acpl.json -sb logs/pretrain -t pretrain_miniimagenet_wrn_acpl
```
## 4.2.Finetune (single-domain example)
```
python train_runner.py configs/finetune_miniimagenet_wrn_1_acpl.json --ckpt logs/pretrain/pretrain_mini_wrn/checkpoint_best.pth -sb logs/finetune -t miniimagenet_wrn_1_acpl
```
## 4.3.Test (single-domain example)
```
python test_runner.py configs/ssl_large_miniimagenet-wrn-1_acpl.json logs/finetune/miniimagenet_wrn_1_acpl/checkpoint_best.pth -sb logs/ACPL-FSL -s ACPL -t ACPL_miniimagenet_wrn_1
```
## 4.4.Cross-Domain (CDFSL)

# finetune
```
python train_runner.py configs/cdfsl/finetune_cdfsl_isic_resnet10_1shot.json \
  --ckpt logs/pretrain/pretrain_miniimagenet_resnet10/checkpoint_best.pth \
  -sb logs/cdfsl -t isic_1shot
```
# test
```
python test_runner.py configs/cdfsl/ssl_isic_resnet10_1shot.json \
  logs/cdfsl/isic_1shot/checkpoint_best.pth \
  -sb logs/cdfsl_results -t isic_1shot_test
```"# ACPL-FSL" 
"# ACPL-FSL" 
"# ACPL-FSL" 
