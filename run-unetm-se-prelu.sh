
for id in {1..5..1}
do
    file_name=unetm-se-prelu-$id
    echo "Id: $id | Log file: logs/train/$file_name.txt"
    python train.py --bands=51 --hsi_c=rad --network_arch=unet --use_cuda --use_mini --use_augs --use_preluSE --use_SE --gamma=$gamma --network_weights_path=savedmodels/$file_name.pt 2>&1 | tee ./logs/save/unetm-se-prelu/train/$file_name.txt 
    python test.py --bands=51 --hsi_c=rad --use_preluSE --use_SE --network_arch=unet --use_cuda --use_mini --network_weights_path=savedmodels/$file_name.pt 2>&1 | tee logs/save/unetm-se-prelu/test/$file_name.txt 
done
