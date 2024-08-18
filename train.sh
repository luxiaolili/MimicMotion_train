CUDA_VISIBLE_DEVICES=0  
  accelerate launch --num_processes 1 --mixed_precision "fp16" train.py \
  --video_folder='video_data' \
  --pretrained_model_name_or_path="stabilityai/stable-video-diffusion-img2vid-xt" \
  --per_gpu_batch_size=1 \
  --max_train_steps=50000 \
  --gradient_checkpointing \
  --width=576 \
  --height=1024 \
  --use_8bit_adam \
  --checkpointing_steps=100 \
  --learning_rate=1e-05 \
  --lr_warmup_steps=0 \
  --seed=123 \