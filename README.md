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





## Training Guidance

1. In the experiments, the posenet is so hard to control, So I do a lot of  experiments for it. I think the posenet is not good for control the pose, But I train the posenet with unet2d, the results shows that posenet can control the pose for sd-2.1, You can follow my other project Pose2Image.(https://github.com/luxiaolili/Pose2Image)
2. The diffusers is unstabitily, I do it with different versions, the result is different
3. It is need clear data and so many datasets, This is a data hungry task
4. It is bad for train many epochs, mybe my dataset is so poor
5. Maybe you should train the posenet on image and finetune the unet and posenet for SVD. (https://github.com/luxiaolili/Pose2Image)
   

### Environment setup

Recommend python 3+ with torch 2.x are validated with an Nvidia A800 GPU. Follow the command below to install all the dependencies of python:

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
```

### dataset structure
```
ubc_data
|-- videos
|-- pose_score
|-- ref
|-- dwpose
```
You can run the script to get the pose, pose_score, reference face pic 
```
python get_video_pose.py ubc_data/videos  dwpose
python get_video_pose_score.py ubc_data/videos  pose_score
python get_video_reference.py ubc_data/videos  ref
```
### Model train

```
sh train.sh
```
or 
```
CUDA_VISIBLE_DEVICES=0  
  accelerate launch --num_processes 1 --mixed_precision "fp16" train.py \
  --video_folder='ubc_data' \
  --pretrained_model_name_or_path="stabilityai/stable-video-diffusion-img2vid-xt-1-1" \
  --per_gpu_batch_size=1 \
  --max_train_steps=50000 \
  --width=576 \
  --height=768 \
  --checkpointing_steps=200 \
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
