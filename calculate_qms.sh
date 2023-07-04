for drug in "Binimetinib" "Blebbistatin" "CK666" "DMSO" "H1152" "MK1775" "Nocodazole" "PF228" "Palbociclib"
do
    python calculate_qms.py --drug $drug --csv_dir "/run/user/1128299809/gvfs/smb-share:server=rds.icr.ac.uk,share=data/DBI/DUDBI/DYNCESYS/mvries/Datasets/VickyCellshape/cn_allFeatures_withGeneNames_updated.csv" --img_dir "/mnt/nvme0n1/Datasets/VickyPlates_010922/TransformerFeats_GEFGAP/"
done
