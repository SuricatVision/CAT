#!/usr/bin/env bash
# model_dir=/media/dereyly/data_hdd/models/gan/pix2pix/metaf-512-mobile3
# python onnx_export.py --dataroot database/maps \
#   --distiller inception \
#   --log_dir $model_dir \
#   --restore_teacher_G_path $model_dir/latest_net_G.pth \
#   --pretrained_student_G_path $model_dir/latest_net_G.pth \
#   --real_stat_path real_stat/cityscapes_A.npz \
#   --teacher_netG inception_9blocks --student_netG inception_9blocks \
#   --pretrained_ngf 64 --teacher_ngf 64 --student_ngf 64 \
#   --norm batch \
#   --norm_affine \
#   --norm_affine_D \
#   --norm_track_running_stats \
#   --channels_reduction_factor 6 \
#   --kernel_sizes 1 3 5 \
#   --direction BtoA \
#   --batch_size 8 \
#   --eval_batch_size 2 \
#   --gpu_ids 0 \
#   --num_threads 8 \
#   --prune_cin_lb 16 \
#   --target_flops 4.6e9 \

model_dir=/media/dereyly/data_hdd/models/gan/pix2pix/metaf-512-mobile3
python onnx_exporter_simple.py  \
  --distiller inception \
  --log_dir $model_dir \
  --pretrained_student_G_path $model_dir/latest_net_G.pth \
  --student_netG inception_9blocks \
  --pretrained_ngf 64 --student_ngf 64 \
  --norm batch \
  --norm_affine \
  --norm_affine_D \
  --norm_track_running_stats \
  --channels_reduction_factor 6 \
  --kernel_sizes 1 3 5 \
  --batch_size 8 \
  --eval_batch_size 2 \
  --gpu_ids 0 \
  --num_threads 8 \
  --prune_cin_lb 16 \
  --target_flops 4.6e9 \
  --data_height 512 --data_width 512