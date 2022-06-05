#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=1
python train.py --dataroot /home/nsergievskiy/ImageDB/metaf \
  --model spade \
  --log_dir logs/pix2pix/metaf2/inception/teacher128b \
  --netG inception_spade \
  --batch_size 16 \
  --nepochs 500 --nepochs_decay 1000 \
  --num_threads 16 \
  --gpu_ids 0 \
  --norm batch \
  --norm_affine \
  --norm_affine_D \
  --norm_track_running_stats \
  --channels_reduction_factor 6 \
  --kernel_sizes 1 3 5 \
  --save_epoch_freq 10 --save_latest_freq 10000 \
  --eval_batch_size 16 \
  --direction AtoB \
  --dataset_mode=aligned \
  --ngf=128  --no_fid  --no_mIoU
  # --restore_pretrained_G_path logs/pix2pix/metaf2/inception/teacher/checkpoints/60_net_G.pth \
  # --restore_D_path logs/pix2pix/metaf2/inception/teacher/checkpoints/60_net_D.pth  
    # --recon_loss_type=vgg \
