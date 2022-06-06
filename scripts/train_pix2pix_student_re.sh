#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=1
python distill.py --dataroot /home/nsergievskiy/ImageDB/metaf \
  --distiller inception \
  --log_dir logs/pix2pix/metaf2/inception/student_64_vgg/4p6B \
  --restore_teacher_G_path /home/nsergievskiy/progs/CAT/logs/pix2pix/metaf2/inception/teacher128_vgg/checkpoints/latest_net_G.pth \
  --nepochs 500 --nepochs_decay 1000 \
  --teacher_netG inception_9blocks --student_netG inception_9blocks \
  --pretrained_ngf 128 --teacher_ngf 128 --student_ngf 64 \
  --real_stat_path /home/nsergievskiy/ImageDB/metaf/images_b.npz \
  --num_threads 20 \
  --eval_batch_size 2 \
  --batch_size 40 \
  --gpu_ids 0 \
  --norm batch \
  --norm_affine \
  --norm_affine_D \
  --norm_track_running_stats \
  --channels_reduction_factor 6 \
  --kernel_sizes 1 3 5 \
  --direction AtoB \
  --lambda_distill 1.3 \
  --prune_cin_lb 16 \
  --target_flops 10.6e9 \
  --distill_G_loss_type ka \
  --recon_loss_type=vgg  --lambda_recon 10 \
  --prune_continue \
  --pretrained_student_G_path=/home/nsergievskiy/progs/CAT/logs/pix2pix/metaf2/inception/student64/4p6B/checkpoints/155_net_G.pth \
  --restore_D_path /home/nsergievskiy/progs/CAT/logs/pix2pix/metaf2/inception/student64/4p6B/checkpoints/155_net_D.pth 
  # --restore_O_path=/home/nsergievskiy/progs/CAT/logs/pix2pix/metaf2/inception/student64/4p6B/checkpoints/150_optim \

  #--prune_continue 

  #  --restore_pretrained_G_path /home/nsergievskiy/progs/CAT/logs/pix2pix/metaf2/inception/teacher128_vgg/checkpoints/latest_net_G.pth \
  # --load_size=256

#--pretrained_student_G_path
#restore_pretrained_G_path
# export CUDA_VISIBLE_DEVICES=1,2
# python train.py --dataroot /home/nsergievskiy/ImageDB/metaf \
#   --model pix2pix \
#   --log_dir logs/pix2pix/metaf2/inception/teacher \
#   --netG inception_9blocks \
#   --batch_size 16 \
#   --lambda_recon 10 \
#   --nepochs 500 --nepochs_decay 1000 \
#   --num_threads 16 \
#   --gpu_ids 0 \
#   --norm batch \
#   --norm_affine \
#   --norm_affine_D \
#   --norm_track_running_stats \
#   --channels_reduction_factor 6 \
#   --kernel_sizes 1 3 5 \
#   --save_epoch_freq 10 --save_latest_freq 10000 \
#   --eval_batch_size 16 \
#   --direction AtoB \
#   --dataset_mode=aligned \
#   --ngf=128
