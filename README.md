# MimicMotion


<div align="center">
<h1>MimicMotion: High-Quality Human Motion Video Generation with Confidence-aware Pose Guidance</h1>
</div>

This is the unofficial train code of MimicMotion: High-Quality Human Motion Video Generation with Confidence-aware Pose Guidance(https://arxiv.org/abs/2406.19680).
## Overview

<p align="center">
  <img src="assets/figures/model_structure.png" alt="model architecture" width="640"/>
  </br>
  <i>An overview of the framework of MimicMotion.</i>
</p>

In recent years, generative artificial intelligence has achieved significant advancements in the field of image generation, spawning a variety of applications. However, video generation still faces considerable challenges in various aspects such as controllability, video length, and richness of details, which hinder the application and popularization of this technology. In this work, we propose a controllable video generation framework, dubbed *MimicMotion*, which can generate high-quality videos of arbitrary length with any motion guidance. Comparing with previous methods, our approach has several highlights. Firstly, with confidence-aware pose guidance, temporal smoothness can be achieved so model robustness can be enhanced with large-scale training data. Secondly, regional loss amplification based on pose confidence significantly eases the distortion of image significantly. Lastly, for generating long smooth videos, a progressive latent fusion strategy is proposed. By this means, videos of arbitrary length can be generated with acceptable resource consumption. With extensive experiments and user studies, MimicMotion demonstrates significant improvements over previous approaches in multiple aspects.



## Quickstart

For the initial released version of the model checkpoint, it supports generating videos with a maximum of 72 frames at a 576x1024 resolution. If you encounter insufficient memory issues, you can appropriately reduce the number of frames.

### Environment setup

Recommend python 3+ with torch 2.x are validated with an Nvidia V100 GPU. Follow the command below to install all the dependencies of python:

```
conda env create -f environment.yaml
conda activate mimicmotion
```

### Download weights
If you experience connection issues with Hugging Face, you can utilize the mirror endpoint by setting the environment variable: `export HF_ENDPOINT=https://hf-mirror.com`.
Please download weights manually as follows:
```
cd MimicMotions/
mkdir models
```
1. Download DWPose pretrained model: [dwpose](https://huggingface.co/yzd-v/DWPose/tree/main)
    ```
    mkdir -p models/DWPose
    wget https://huggingface.co/yzd-v/DWPose/resolve/main/yolox_l.onnx?download=true -O models/DWPose/yolox_l.onnx
    wget https://huggingface.co/yzd-v/DWPose/resolve/main/dw-ll_ucoco_384.onnx?download=true -O models/DWPose/dw-ll_ucoco_384.onnx
    ```
2. Download the pre-trained checkpoint of MimicMotion from [Huggingface](https://huggingface.co/ixaac/MimicMotion)
    ```
    wget -P models/ https://huggingface.co/ixaac/MimicMotion/resolve/main/MimicMotion_1-1.pth
    ```
3. The SVD model [stabilityai/stable-video-diffusion-img2vid-xt-1-1](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt-1-1) will be automatically downloaded.

Finally, all the weights should be organized in models as follows

```
models/
├── DWPose
│   ├── dw-ll_ucoco_384.onnx
│   └── yolox_l.onnx
└── MimicMotion_1-1.pth
```

### train data structure
```
video_data/XXXvideo
|-- imgs
|-- faces
|-- lmks
```

### Model train

```
sh train.sh
```
or 
```
CUDA_VISIBLE_DEVICES=0  
  accelerate launch --num_processes 1 --mixed_precision "fp16" train.py \
  --video_folder='video_data' \
  --pretrained_model_name_or_path="stabilityai/stable-video-diffusion-img2vid-xt" \
  --per_gpu_batch_size=1 \
  --max_train_steps=50000 \
  --width=576 \
  --height=1024 \
  --use_8bit_adam \
  --checkpointing_steps=100 \
  --learning_rate=1e-05 \
  --lr_warmup_steps=0 \
  --seed=123 \
```


### Model inference

A sample configuration for testing is provided as `test.yaml`. You can also easily modify the various configurations according to your needs.

```
python inference.py --inference_config configs/test.yaml
```

Tips: if your GPU memory is limited, try set env `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256`.

### VRAM requirement and Runtime

For the 35s demo video, the 72-frame model requires 16GB VRAM (4060ti) and finishes in 20 minutes on a 4090 GPU.

The minimum VRAM requirement for the 16-frame U-Net model is 8GB; however, the VAE decoder demands 16GB. You have the option to run the VAE decoder on CPU.

## Citation	
```bib
@article{mimicmotion2024,
  title={MimicMotion: High-Quality Human Motion Video Generation with Confidence-aware Pose Guidance},
  author={Yuang Zhang and Jiaxi Gu and Li-Wen Wang and Han Wang and Junqi Cheng and Yuefeng Zhu and Fangyuan Zou},
  journal={arXiv preprint arXiv:2406.19680},
  year={2024}
}
```
