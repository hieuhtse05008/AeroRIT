
CUDA_VISIBLE_DEVICES=0


for gamma in {0..8..1}
do
    echo "$gamma"
    for id in {1..5..1}
    do
        file_name=focal-loss-$gamma-$id
        echo "Gamma = $gamma | Id: $id | Log file: logs/train/$file_name.txt"
        python3 train.py --bands=51 --hsi_c=rad --network_arch=unet --use_cuda --use_mini --use_augs --use_preluSE --use_SE --gamma=$gamma --network_weights_path=savedmodels/$file_name.pt 2>&1 | tee ./logs/train/$file_name.txt 
        python3 test.py --bands=51 --hsi_c=rad --use_preluSE --use_SE --network_arch=unet --use_cuda --use_mini --network_weights_path=savedmodels/$file_name.pt 2>&1 | tee logs/test/$file_name.txt 
    done
    
done
