#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=1
python train.py --dataroot /home/nsergievskiy/ImageDB/metaf \
  --model pix2pix \
  --log_dir logs/pix2pix/metaf2/inception/teacher128_vgg_2 \
  --netG inception_9blocks \
  --batch_size 16 \
  --lambda_recon 10 \
  --nepochs 500 --nepochs_decay 1000 \
  --num_threads 16 \
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
  --recon_loss_type=vgg --load_size=256 \
  --restore_G_path logs/pix2pix/metaf2/inception/teacher128_vgg/checkpoints/latest_net_G.pth \
  --restore_D_path logs/pix2pix/metaf2/inception/teacher128_vgg/checkpoints/latest_net_D.pth