#!/usr/bin/env bash
python train.py --dataroot /media/dereyly/svision_ssd2/Dropbox/datasets/data_jpg/ffhq-metfaces_blended \
  --model pix2pix \
  --log_dir logs/pix2pix/metaf2/inception/teacher \
  --netG inception_9blocks \
  --batch_size 32 \
  --lambda_recon 10 \
  --nepochs 500 --nepochs_decay 1000 \
  --num_threads 32 \
  --gpu_ids 0,1 \
  --norm batch \
  --norm_affine \
  --norm_affine_D \
  --norm_track_running_stats \
  --channels_reduction_factor 6 \
  --kernel_sizes 1 3 5 \
  --save_epoch_freq 10 --save_latest_freq 10000 \
  --eval_batch_size 16 \
  --direction AtoB \
  --dataset_mode=aligned 
