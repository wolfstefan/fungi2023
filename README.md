# Transformer-based Fine-Grained Fungi Classification in an Open-Set Scenario

This repository is targeted towards solving the FungiCLEF 2023 (https://huggingface.co/spaces/competitions/FungiCLEF2023) challenge. It is based on MMPreTrain (https://github.com/open-mmlab/mmpretrain).

## Usage

### Installation

```bash
conda create -n fungi2023 python=3.10 pytorch=2.0.1 torchvision=0.15.2 pytorch-cuda=11.8 -c pytorch -c nvidia
conda activate fungi2023
pip install -r requirements.txt
mim install "mmpretrain==1.0.0rc7"
```

### Data

The challenge data has to be downloaded and put into _data/fungiclef2022/_.

### Training

```bash
bash tools/dist_train.sh configs/swinv2_base_w24_b32x4-fp16_fungi+val_res_384_cb_epochs_6.py 4
```

### Inference on pre-trained models

```bash
python tools/test_generate_result_pre-consensus_tta.py models/swinv2_base_w24_b32x4-fp16_fungi+val_res_384_cb_epochs_6.py models/swinv2_base_w24_b32x4-fp16_fungi+val_res_384_cb_epochs_6_20230524-a251a50a.pth results.csv --threshold 0.2 --no-scores
```
