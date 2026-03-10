# Multi-Scale Adaptive Clustering and Local Consistency Learning for Unsupervised Clothing-Changing Person Re-Identification

[[paper]](https://ieeexplore.ieee.org/document/11422274)

## Upload History

* 2026/03/10: Upload Code and Weight.

## Installation

Install `conda` before installing any requirements.

```bash
conda create -n malc python=3.8
conda activate malc
pip install -r requirements.txt
```

## Datasets
We evaluate our method on the following four public clothing-changing Re-ID datasets:

* **PRCC**
* **LTCC**
* **Celeb-reID**
* **Celeb-reID-light**


## Weights

You can download the pre-trained weights from the following link:

* **[Download Best Weights (Google Drive)](https://drive.google.com/drive/folders/1u8UtrBGJptN6kJhOYtfKqWaXpWMWKrSj?usp=sharing)**
* Includes: `prcc_best.pth.tar`, `ltcc_best.pth.tar`, `celeb_best.pth.tar`, `celeb_light_best.pth.tar`

## Training

For example, training the full model on prcc with GPU 0 and saving the log file and checkpoints to `logs/prcc`:

```
CUDA_VISIBLE_DEVICES=0 python train.py -b 128 -d ltcc --iters 200 --eps 0.5 --data-dir /home/ykding/dataset --logs-dir ./log/ltcc

CUDA_VISIBLE_DEVICES=0 python train.py -b 128 -d prcc --iters 200 --eps 0.5 --data-dir /home/ykding/dataset --logs-dir ./log/prcc

CUDA_VISIBLE_DEVICES=0 python train.py -b 128 -d celebreid --iters 200 --eps 0.5 --data-dir /home/ykding/dataset --logs-dir ./log/celeb

CUDA_VISIBLE_DEVICES=0 python train.py -b 128 -d celebreidlight --iters 200 --eps 0.5 --data-dir /home/ykding/dataset --logs-dir ./log/celeb-light

```

## Test
```
CUDA_VISIBLE_DEVICES=0 python inference_clip_cc.py -d prcc --data-dir /home/ykding/dataset --checkpoint /path/to/model_best.pth.tar
```

## Loss Parameter Adjustment

If you need to adjust the loss hyper-parameters, please refer to the trainer file located at `clip_cc/trainers.py`.

For instance, around **line 73**, you can modify the weights for different components. The optimal parameters for the **LTCC** dataset are `1.0` and `0.7`:

```python
# clip_cc/trainers.py (Line 73)
loss = 1.0 * loss_global + 0.7 * loss_local
```

[//]: # (## Results)

[//]: # (![Experimental Results]&#40;images/result.png&#41;)



[//]: # (## Note)

[//]: # ()
[//]: # (The code is implemented based on following works.)

[//]: # (1. [PCL-CLIP]&#40;https://github.com/RikoLi/PCL-CLIP&#41;)

[//]: # (2. [ClusterContrast]&#40;https://github.com/alibaba/cluster-contrast-reid&#41;)



