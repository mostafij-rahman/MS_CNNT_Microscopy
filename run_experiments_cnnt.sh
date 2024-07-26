
#python3 main.py --load_path logs/model/backbone_training_all_data_mscnnt_k13_ps_pwc_mlp11_loss_ssim1_charbonnier1_tv_val_ns8_bs24_epoch100_run_0__07-23-2024_T21-19-03_epoch-100_best.pt --fine_samples 1 --h5files /scratch/micro_datasets_tvt_split/micro_datasets_tvt_split/Chris_zebra_new_train.h5 --val_case /scratch/micro_datasets_tvt_split/micro_datasets_tvt_split/Chris_zebra_new_val.h5 --test_case /scratch/micro_datasets_tvt_split/micro_datasets_tvt_split/Chris_zebra_new_test.h5 --ratio 100 0 0 --global_lr 0.000025 --num_epochs 100 --batch_size 8 --time 16 --width 128 160 --height 128 160 --skip_LSUV --loss ssim charbonnier --loss_weights 1.0 1.0 --im_value_scale 0 4096 --wandb_entity gadgetron --run_name fine_tuning_1shot_Chris_zebra_all_data_mscnnt_k13_ps_pwc_mlp11_loss_ssim1_charbonnier1_tv_val_ns8_bs8_epoch100_run_0 --run_notes fine_tuning_1shot_Chris_zebra_all_data_mscnnt_k13_ps_pwc_mlp11_loss_ssim1_charbonnier1_tv_val_ns8_bs8_100_epochs

#python3 main.py --load_path logs/model/backbone_training_all_data_mscnnt_k13_ps_pwc_mlp11_loss_ssim1_charbonnier1_tv_val_ns8_bs24_epoch100_run_0__07-23-2024_T21-19-03_epoch-100_best.pt --fine_samples 5 --h5files /scratch/micro_datasets_tvt_split/micro_datasets_tvt_split/Chris_zebra_new_train.h5 --val_case /scratch/micro_datasets_tvt_split/micro_datasets_tvt_split/Chris_zebra_new_val.h5 --test_case /scratch/micro_datasets_tvt_split/micro_datasets_tvt_split/Chris_zebra_new_test.h5 --ratio 100 0 0 --global_lr 0.000025 --num_epochs 100 --batch_size 8 --time 16 --width 128 160 --height 128 160 --skip_LSUV --loss ssim charbonnier --loss_weights 1.0 1.0 --im_value_scale 0 4096 --wandb_entity gadgetron --run_name fine_tuning_5shots_Chris_zebra_all_data_mscnnt_k13_ps_pwc_mlp11_loss_ssim1_charbonnier1_tv_val_ns8_bs8_epoch100_run_0 --run_notes fine_tuning_5shots_Chris_zebra_all_data_mscnnt_k13_ps_pwc_mlp11_loss_ssim1_charbonnier1_tv_val_ns8_bs8_100_epochs

#python3 main.py --load_path logs/model/backbone_training_isim_mscnnt_k13_ps_pwc_mlp11_loss_ssim1_charbonnier1_tv_val_ns8_bs24_epoch100_run_0__07-24-2024_T14-10-29_epoch-100_best.pt --fine_samples 1 --h5files /scratch/micro_datasets_tvt_split/micro_datasets_tvt_split/Chris_zebra_new_train.h5 --val_case /scratch/micro_datasets_tvt_split/micro_datasets_tvt_split/Chris_zebra_new_val.h5 --test_case /scratch/micro_datasets_tvt_split/micro_datasets_tvt_split/Chris_zebra_new_test.h5 --ratio 100 0 0 --global_lr 0.000025 --num_epochs 100 --batch_size 8 --time 16 --width 128 160 --height 128 160 --skip_LSUV --loss ssim charbonnier --loss_weights 1.0 1.0 --im_value_scale 0 4096 --wandb_entity gadgetron --run_name fine_tuning_1shot_Chris_zebra_isim_mscnnt_k13_ps_pwc_mlp11_loss_ssim1_charbonnier1_tv_val_ns8_bs8_epoch100_run_0 --run_notes fine_tuning_1shot_Chris_zebra_isim_mscnnt_k13_ps_pwc_mlp11_loss_ssim1_charbonnier1_tv_val_ns8_bs8_100_epochs

#python3 main.py --load_path logs/model/backbone_training_isim_mscnnt_k13_ps_pwc_mlp11_loss_ssim1_charbonnier1_tv_val_ns8_bs24_epoch100_run_0__07-24-2024_T14-10-29_epoch-100_best.pt --fine_samples 5 --h5files /scratch/micro_datasets_tvt_split/micro_datasets_tvt_split/Chris_zebra_new_train.h5 --val_case /scratch/micro_datasets_tvt_split/micro_datasets_tvt_split/Chris_zebra_new_val.h5 --test_case /scratch/micro_datasets_tvt_split/micro_datasets_tvt_split/Chris_zebra_new_test.h5 --ratio 100 0 0 --global_lr 0.000025 --num_epochs 100 --batch_size 8 --time 16 --width 128 160 --height 128 160 --skip_LSUV --loss ssim charbonnier --loss_weights 1.0 1.0 --im_value_scale 0 4096 --wandb_entity gadgetron --run_name fine_tuning_5shots_Chris_zebra_isim_mscnnt_k13_ps_pwc_mlp11_loss_ssim1_charbonnier1_tv_val_ns8_bs8_epoch100_run_0 --run_notes fine_tuning_5shots_Chris_zebra_isim_mscnnt_k13_ps_pwc_mlp11_loss_ssim1_charbonnier1_tv_val_ns8_bs8_100_epochs

#python3 main.py --load_path logs/model/backbone_training_isim_mscnnt_k13_ps_pwc_mlp11_loss_ssim1_charbonnier1_tv_val_ns8_bs24_epoch100_run_0__07-24-2024_T14-10-29_epoch-100_best.pt --fine_samples 1 --h5files /scratch/micro_datasets_tvt_split/micro_datasets_tvt_split/Alex_wide_field_new_train.h5 --val_case /scratch/micro_datasets_tvt_split/micro_datasets_tvt_split/Alex_wide_field_new_val.h5 --test_case /scratch/micro_datasets_tvt_split/micro_datasets_tvt_split/Alex_wide_field_new_test.h5 --ratio 100 0 0 --global_lr 0.000025 --num_epochs 100 --batch_size 8 --time 16 --width 128 160 --height 128 160 --skip_LSUV --loss ssim charbonnier --loss_weights 1.0 1.0 --im_value_scale 0 4096 --wandb_entity gadgetron --run_name fine_tuning_1shot_Alex_wide_isim_mscnnt_k13_ps_pwc_mlp11_loss_ssim1_charbonnier1_tv_val_ns8_bs8_epoch100_run_0 --run_notes fine_tuning_1shot_Alex_wide_isim_mscnnt_k13_ps_pwc_mlp11_loss_ssim1_charbonnier1_tv_val_ns8_bs8_100_epochs

#python3 main.py --load_path logs/model/backbone_training_isim_mscnnt_k13_ps_pwc_mlp11_loss_ssim1_charbonnier1_tv_val_ns8_bs24_epoch100_run_0__07-24-2024_T14-10-29_epoch-100_best.pt --fine_samples 5 --h5files /scratch/micro_datasets_tvt_split/micro_datasets_tvt_split/Alex_wide_field_new_train.h5 --val_case /scratch/micro_datasets_tvt_split/micro_datasets_tvt_split/Alex_wide_field_new_val.h5 --test_case /scratch/micro_datasets_tvt_split/micro_datasets_tvt_split/Alex_wide_field_new_test.h5 --ratio 100 0 0 --global_lr 0.000025 --num_epochs 100 --batch_size 8 --time 16 --width 128 160 --height 128 160 --skip_LSUV --loss ssim charbonnier --loss_weights 1.0 1.0 --im_value_scale 0 4096 --wandb_entity gadgetron --run_name fine_tuning_5shots_Alex_wide_isim_mscnnt_k13_ps_pwc_mlp11_loss_ssim1_charbonnier1_tv_val_ns8_bs8_epoch100_run_0 --run_notes fine_tuning_5shots_Alex_wide_isim_mscnnt_k13_ps_pwc_mlp11_loss_ssim1_charbonnier1_tv_val_ns8_bs8_100_epochs

#python3 main.py --load_path logs/model/backbone_training_isim_mscnnt_k13_ps_pwc_mlp11_loss_ssim1_charbonnier1_tv_val_ns8_bs24_epoch100_run_0__07-24-2024_T14-10-29_epoch-100_best.pt --fine_samples 1 --h5files /scratch/micro_datasets_tvt_split/micro_datasets_tvt_split/Alex_wide_field_new_train.h5 --val_case /scratch/micro_datasets_tvt_split/micro_datasets_tvt_split/Alex_wide_field_new_val.h5 --test_case /scratch/micro_datasets_tvt_split/micro_datasets_tvt_split/Alex_wide_field_new_test.h5 --ratio 100 0 0 --global_lr 0.000025 --num_epochs 100 --batch_size 8 --time 16 --width 128 160 --height 128 160 --skip_LSUV --loss ssim charbonnier --loss_weights 1.0 1.0 --im_value_scale 0 4096 --wandb_entity gadgetron --run_name fine_tuning_1shot_Alex_wide_isim_mscnnt_k13_ps_pwc_mlp11_loss_ssim1_charbonnier1_tv_val_ns8_bs8_epoch100_run_0 --run_notes fine_tuning_1shot_Alex_wide_isim_mscnnt_k13_ps_pwc_mlp11_loss_ssim1_charbonnier1_tv_val_ns8_bs8_100_epochs

#python3 main.py --load_path logs/model/backbone_training_all_data_mscnnt_k13_ps_pwc_mlp11_loss_ssim1_charbonnier1_tv_val_ns8_bs24_epoch100_run_0__07-23-2024_T21-19-03_epoch-100_best.pt --fine_samples 1 --h5files /scratch/micro_datasets_tvt_split/micro_datasets_tvt_split/Alex_wide_field_new_train.h5 --val_case /scratch/micro_datasets_tvt_split/micro_datasets_tvt_split/Alex_wide_field_new_val.h5 --test_case /scratch/micro_datasets_tvt_split/micro_datasets_tvt_split/Alex_wide_field_new_test.h5 --ratio 100 0 0 --global_lr 0.000025 --num_epochs 100 --batch_size 8 --time 16 --width 128 160 --height 128 160 --skip_LSUV --loss ssim charbonnier --loss_weights 1.0 1.0 --im_value_scale 0 4096 --wandb_entity gadgetron --run_name fine_tuning_1shot_Alex_wide_all_data_mscnnt_k13_ps_pwc_mlp11_loss_ssim1_charbonnier1_tv_val_ns8_bs8_epoch100_run_0 --run_notes fine_tuning_1shot_Alex_wide_all_data_mscnnt_k13_ps_pwc_mlp11_loss_ssim1_charbonnier1_tv_val_ns8_bs8_100_epochs

#python3 main.py --load_path logs/model/backbone_training_all_data_mscnnt_k13_ps_pwc_mlp11_loss_ssim1_charbonnier1_tv_val_ns8_bs24_epoch100_run_0__07-23-2024_T21-19-03_epoch-100_best.pt --fine_samples 5 --h5files /scratch/micro_datasets_tvt_split/micro_datasets_tvt_split/Alex_wide_field_new_train.h5 --val_case /scratch/micro_datasets_tvt_split/micro_datasets_tvt_split/Alex_wide_field_new_val.h5 --test_case /scratch/micro_datasets_tvt_split/micro_datasets_tvt_split/Alex_wide_field_new_test.h5 --ratio 100 0 0 --global_lr 0.000025 --num_epochs 100 --batch_size 8 --time 16 --width 128 160 --height 128 160 --skip_LSUV --loss ssim charbonnier --loss_weights 1.0 1.0 --im_value_scale 0 4096 --wandb_entity gadgetron --run_name fine_tuning_5shots_Alex_wide_all_data_mscnnt_k13_ps_pwc_mlp11_loss_ssim1_charbonnier1_tv_val_ns8_bs8_epoch100_run_0 --run_notes fine_tuning_5shots_Alex_wide_all_data_mscnnt_k13_ps_pwc_mlp11_loss_ssim1_charbonnier1_tv_val_ns8_bs8_100_epochs

#python3 main.py --load_path logs/model/backbone_training_all_data_mscnnt_k13_ps_pwc_mlp11_loss_ssim1_charbonnier1_tv_val_ns8_bs24_epoch100_run_0__07-23-2024_T21-19-03_epoch-100_best.pt --fine_samples 1 --h5files /scratch/micro_datasets_tvt_split/micro_datasets_tvt_split/Ryo_tile_new_ft_train.h5 --val_case /scratch/micro_datasets_tvt_split/micro_datasets_tvt_split/Ryo_tile_new_val.h5 --test_case /scratch/micro_datasets_tvt_split/micro_datasets_tvt_split/Ryo_tile_new_test.h5 --ratio 100 0 0 --global_lr 0.000025 --num_epochs 100 --batch_size 8 --time 16 --width 128 160 --height 128 160 --skip_LSUV --loss ssim charbonnier --loss_weights 1.0 1.0 --im_value_scale 0 4096 --wandb_entity gadgetron --run_name fine_tuning_1shot_Ryo_tile_all_data_mscnnt_k13_ps_pwc_mlp11_loss_ssim1_charbonnier1_tv_val_ns8_bs8_epoch100_run_0 --run_notes fine_tuning_1shot_Ryo_tile_all_data_mscnnt_k13_ps_pwc_mlp11_loss_ssim1_charbonnier1_tv_val_ns8_bs8_100_epochs

#python3 main.py --load_path logs/model/backbone_training_isim_mscnnt_k13_ps_pwc_mlp11_loss_ssim1_charbonnier1_tv_val_ns8_bs24_epoch100_run_0__07-24-2024_T14-10-29_epoch-100_best.pt --fine_samples 1 --h5files /scratch/micro_datasets_tvt_split/micro_datasets_tvt_split/Ryo_tile_new_ft_train.h5 --val_case /scratch/micro_datasets_tvt_split/micro_datasets_tvt_split/Ryo_tile_new_val.h5 --test_case /scratch/micro_datasets_tvt_split/micro_datasets_tvt_split/Ryo_tile_new_test.h5 --ratio 100 0 0 --global_lr 0.000025 --num_epochs 100 --batch_size 8 --time 16 --width 128 160 --height 128 160 --skip_LSUV --loss ssim charbonnier --loss_weights 1.0 1.0 --im_value_scale 0 4096 --wandb_entity gadgetron --run_name fine_tuning_1shot_Ryo_tile_isim_mscnnt_k13_ps_pwc_mlp11_loss_ssim1_charbonnier1_tv_val_ns8_bs8_epoch100_run_0 --run_notes fine_tuning_1shot_Ryo_tile_isim_mscnnt_k13_ps_pwc_mlp11_loss_ssim1_charbonnier1_tv_val_ns8_bs8_100_epochs

#python3 main.py --load_path logs/model/backbone_training_all_data_mscnnt_k13_ps_pwc_mlp11_loss_ssim1_charbonnier1_tv_val_ns8_bs24_epoch100_run_0__07-23-2024_T21-19-03_epoch-100_best.pt --fine_samples 5 --h5files /scratch/micro_datasets_tvt_split/micro_datasets_tvt_split/Ryo_tile_new_ft_train.h5 --val_case /scratch/micro_datasets_tvt_split/micro_datasets_tvt_split/Ryo_tile_new_val.h5 --test_case /scratch/micro_datasets_tvt_split/micro_datasets_tvt_split/Ryo_tile_new_test.h5 --ratio 100 0 0 --global_lr 0.000025 --num_epochs 100 --batch_size 8 --time 16 --width 128 160 --height 128 160 --skip_LSUV --loss ssim charbonnier --loss_weights 1.0 1.0 --im_value_scale 0 4096 --wandb_entity gadgetron --run_name fine_tuning_5shots_Ryo_tile_all_data_mscnnt_k13_ps_pwc_mlp11_loss_ssim1_charbonnier1_tv_val_ns8_bs8_epoch100_run_0 --run_notes fine_tuning_5shots_Ryo_tile_all_data_mscnnt_k13_ps_pwc_mlp11_loss_ssim1_charbonnier1_tv_val_ns8_bs8_100_epochs

#python3 main.py --load_path logs/model/backbone_training_isim_mscnnt_k13_ps_pwc_mlp11_loss_ssim1_charbonnier1_tv_val_ns8_bs24_epoch100_run_0__07-24-2024_T14-10-29_epoch-100_best.pt --fine_samples 5 --h5files /scratch/micro_datasets_tvt_split/micro_datasets_tvt_split/Ryo_tile_new_ft_train.h5 --val_case /scratch/micro_datasets_tvt_split/micro_datasets_tvt_split/Ryo_tile_new_val.h5 --test_case /scratch/micro_datasets_tvt_split/micro_datasets_tvt_split/Ryo_tile_new_test.h5 --ratio 100 0 0 --global_lr 0.000025 --num_epochs 100 --batch_size 8 --time 16 --width 128 160 --height 128 160 --skip_LSUV --loss ssim charbonnier --loss_weights 1.0 1.0 --im_value_scale 0 4096 --wandb_entity gadgetron --run_name fine_tuning_5shots_Ryo_tile_isim_mscnnt_k13_ps_pwc_mlp11_loss_ssim1_charbonnier1_tv_val_ns8_bs8_epoch100_run_0 --run_notes fine_tuning_5shots_Ryo_tile_isim_mscnnt_k13_ps_pwc_mlp11_loss_ssim1_charbonnier1_tv_val_ns8_bs8_100_epochs

