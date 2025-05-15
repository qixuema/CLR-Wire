
# CLR-Wire: Towards Continuous Latent Representations for 3D Curve Wireframe Generation (ACM SIGGRAPH 2025)

<a href="https://pytorch.org/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white"></a>
<a href="https://github.com/huggingface/accelerate"><img alt="Accelerate" src="https://img.shields.io/badge/Accelerate-ffd21e?style=for-the-badge&logo=Accelerate&logoColor=white"></a>
<a href="https://wandb.ai/site"><img alt="WandB" src="https://img.shields.io/badge/Weights_&_Biases-FFBE00?style=for-the-badge&logo=WeightsAndBiases&logoColor=white"></a>

**This repository is the official repository of the paper, *CLR-Wire: Towards Continuous Latent Representations for 3D Curve Wireframe Generation*.**

[Xueqi Ma](https://qixuema.github.io/),
[Yilin Liu](https://yilinliu77.github.io/),
[Tianlong Gao](https://github.com/Alone-gao/),
[Qirui Huang](https://scholar.google.com/citations?user=PWoil2gAAAAJ&hl=zh-CN),
[Hui Huang](https://vcc.tech/~huihuang),

[VCC](https://vcc.tech/),
[CSSE](https://csse.szu.edu.cn/),
[Shenzhen University](https://www.szu.edu.cn/)


### [Project Page](https://vcc.tech/research/2025/CLRWire) | [Paper (ArXiv)](https://www.arxiv.org/abs/2504.19174)


<img src='assets/teaser.png'/>

https://github.com/user-attachments/assets/adc120d7-6ac1-45fd-bcbb-b6081ddf3cd0


## Installation
The code is tested in docker enviroment [nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04](https://hub.docker.com/layers/nvidia/cuda/12.4.1-cudnn-devel-ubuntu22.04/images/sha256-0a1cb6e7bd047a1067efe14efdf0276352d5ca643dfd77963dab1a4f05a003a4).
The following are instructions for setting up the environment in a Linux system from scratch.

First, clone this repository:

      git clone git@github.com:qixuema/CLR-Wire.git

Then, create a mamba environment with the yaml file. (Sometimes the conda is a bit slow to solve the dependencies, so [mamba](https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html) is recommended). You could also just use conda as well.

      mamba env create -f environment.yml
      mamba activate wire

## Download data and checkpoints

We use the [ABC](https://deep-geometry.github.io/abc-dataset/) dataset and process its shapes into curve wireframe (aka. curve network). The processed data is stored on [huggingface](https://huggingface.co/datasets/qixuema/CurveWiframe).

## Usage

To train the model, please use the following commands:

      # Train CurveVAE
      accelerate launch --config_file src/configs/default_config.yaml train_curve_vae.py --config src/configs/train_curve_vae.yaml

      # Train WireframeVAE
      accelerate launch --config_file src/configs/default_config.yaml train_wireframe_vae.py --config src/configs/train_wireframe_vae.yaml

      # Train Flow Matching
      accelerate launch --config_file src/configs/default_config.yaml train_flow_matching.py --config src/configs/train_flow_matching.yaml

## :notebook_with_decorative_cover: Citation

If you find our work useful for your research, please consider citing the following papers :)

```bibtex
@inproceedings{CLRWire25,
    title = {CLR-Wire: Towards Continuous Latent Representations for 3D Curve Wireframe Generation},
    author = {Xueqi Ma and Yilin Liu and Tianlong Gao and Qirui Huang and Hui Huang},
    booktitle = {ACM SIGGRAPH},
    pages = {},
    year = {2025},
}
```

## :email: Contact

This repo is currently maintained by Xueqi Ma ([@qixuema](https://github.com/qixuema)) and is for academic research use only. Discussions and questions are welcome via qixuemaa@gmail.com. 
Layout and styling adapted from [3dlg-hcvc/omages](https://github.com/3dlg-hcvc/omages).