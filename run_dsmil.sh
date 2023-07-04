for fold in 0 1 2 3 4 5 6 7 8 9
do
  for drug in "Binimetinib" "Blebbistatin" "CK666" "H1152" "MK1775" "Nocodazole" "Palbociclib" "PF228"
  do
    python train_transabmil_wandb.py --project_name 'DSMIL' --logger "wandb" --drug $drug --fold $fold --csv_dir "/mnt/nvme0n1/Datasets/SingleCellFromNathan_17122021/folds_3DMIL/all_data_removedwrong_ori_removedTwo_train_test_50_20_30_fold0${fold}.csv"
    done
done
