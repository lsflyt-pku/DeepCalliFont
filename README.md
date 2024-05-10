# DeepCalliFont
This repo is the official implementation of DeepCalliFont: Few-shot Chinese Calligraphy Font Synthesis by Integrating Dual-modality Generative Models (AAAI 2024)

## Prerequisites
* Python
* PyTorch
* Aim

## Data
We provide an example of the dataset in ([PKU Disk](https://disk.pku.edu.cn/link/AA879C32E05E4241319BFE6D4AF2B1C8B0)). You need to unzip it to the "data" directory.

## Train
Run python train.py #path_of_config

example: python train.py config/train_common.yaml

## Test
Run python train.py #path_of_config

example: python test.py config/train_common.yaml
