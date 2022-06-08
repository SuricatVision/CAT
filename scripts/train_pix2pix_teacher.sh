#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=3
python train.py --dataroot /home/nsergievskiy/ImageDB/metaf \
  --model pix2pix \
  --log_dir logs/pix2pix/metaf2/inception/teacher128_vgg_mask2 \
  --real_stat_path /home/nsergievskiy/ImageDB/metaf/images_b.npz \
  --netG inception_9blocks \
  --batch_size 14 \
  --lambda_recon 10 \
  --nepochs 500 --nepochs_decay 1000 \
  --num_threads 7 \
  --gpu_ids 0 \
  --norm batch \
  --norm_affine \
  --norm_affine_D \
  --norm_track_running_stats \
  --channels_reduction_factor 6 \
  --kernel_sizes 1 3 5 \
  --save_epoch_freq 5 --save_latest_freq 10000 \
  --eval_batch_size 16 \
  --direction AtoB \
  --dataset_mode=aligned \
  --ngf=128 \
  --restore_G_path logs/pix2pix/metaf2/inception/teacher128_vgg_mask2/checkpoints/latest_net_G.pth \
  --restore_D_path logs/pix2pix/metaf2/inception/teacher128_vgg_mask2/checkpoints/latest_net_D.pth \
  --recon_loss_type=vgg \
  # 
  #