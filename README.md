# 🚀 RSD-VL: A Relative Semantic Discrimination Vision-Language Model for Anomaly Detection in Autonomous Driving Scenes

This is the official implementation of **RSD-VL: A Relative Semantic Discrimination Vision-Language Model for Anomaly Detection in Autonomous Driving Scenes**.

<p align="center">
  <img src="introduction.png" alt="Introduction" width="700">
</p>

## Abstract

Accurate road anomaly detection is essential for autonomous driving safety, as it identifies unknown obstacles within drivable areas. However, existing methods often rely on visual responses or prediction confidence for anomaly detection, making them vulnerable to visual saliency bias introduced by pseudo-anomalous regions such as shadows, reflections, and lane markings.

To address this problem, we propose **RSD-VL**, a relative semantic discrimination vision-language model for anomaly detection in autonomous driving scenes. RSD-VL constructs a road-specific dual-branch prompt bank to establish a clearer semantic boundary between normal road regions and anomalous obstacles. It further introduces a relative semantic discrimination mechanism that derives anomaly scores by comparing abnormal semantic responses with normal road semantic responses.

In addition, we develop a safety-aware optimization strategy to alleviate irrelevant background interference and reduce the impact of extreme anomaly scores. Experiments on RoadAnomaly, SMIYC-RO21, and Fishyscapes show that RSD-VL achieves strong AP and AuROC while reducing false positives on several datasets.

## Framework

<p align="center">
  <img src="framework.png" alt="Framework" width="900">
</p>

## Installation

Clone this repository:

```bash
git clone https://github.com/Shirui2001/RSD-VL.git
cd RSD-VL
```

Create a conda environment:

```bash
conda create -n rsdvl python=3.9
conda activate rsdvl
```

Install PyTorch 2.3.1 with CUDA 11.8:

```bash
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu118
```

Install other required packages:

```bash
pip install -r requirements.txt
```

## Pretrained Weights

The pretrained checkpoint can be downloaded from the release page:

[Download RSD-VL pretrained weights](https://github.com/Shirui2001/RSD-VL/releases/tag/v1.0)

Please download `rsdvl_best.zip` and unzip it. The archive contains:

```bash
rsdvl_best.pth
```

After extraction, place the checkpoint under:

```bash
checkpoints/rsdvl_best.pth
```

If the `checkpoints/` folder does not exist, please create it manually.

The model definition is in `./model/`. We thank [OpenCLIP](https://github.com/mlfoundations/open_clip) for being open-source. To run the code, please download the OpenCLIP ViT-L-14-336px weights and place them under `./model/`.

## Datasets

We use datasets for inlier training, outlier supervision, and anomaly evaluation. Please prepare the datasets as follows.

- **Inlier Dataset (Cityscapes/Streethazard):** prepare the dataset following the structure provided [here](https://github.com/facebookresearch/Mask2Former/blob/main/datasets/README.md).

- **Outlier Supervision Dataset (MS-COCO):** prepare OOD object annotations using this [COCO preparation script](https://github.com/robin-chan/meta-ood/blob/master/preparation/prepare_coco_segmentation.py), and update `cfg.MODEL.MASK_FORMER.ANOMALY_FILEPATH` accordingly.

- **Anomaly Evaluation Dataset:** download the evaluation datasets from this [link](https://drive.google.com/drive/folders/1eQhmPbKSZrN1AsieY9KFchfll7XC1_SF). Please unzip the files and place them preferably under the `dataset/` folder.

## Training and Evaluation

Training:

```bash
python train.py --shot $shot --save_path $save_path
```

Evaluation:

```bash
python test.py --save_path $save_path --dataset $dataset
```

Optional script for training and evaluation:

```bash
bash scripts.sh
```

## Results

RSD-VL is evaluated on RoadAnomaly, SMIYC-RO21, Fishyscapes Static, and Fishyscapes Lost & Found. The proposed method achieves strong anomaly localization performance and effectively suppresses false positives caused by pseudo-anomalous regions such as shadows, reflections, lane markings, and road textures.

## Acknowledgements

We thank the authors of the following open-source projects:

- [Meta-OOD](https://github.com/robin-chan/meta-ood)
- [Mask2Former](https://github.com/facebookresearch/Mask2Former/tree/main)
- [AA-CLIP](https://github.com/Mwxinnn/AA-CLIP)
- [OpenCLIP](https://github.com/mlfoundations/open_clip)

## Contact

If you have any questions, please contact the authors.
