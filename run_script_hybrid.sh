#!/bin/sh

#srun python _LSGD.py -a resnet50 --epoch 100 --batch-size 64 --gpu-num 4 --lr 6.4 /global/cscratch1/sd/kwangmin/dataset/ImageNet/ILSVRC2012


#srun python LSGD.py --epoch 90 --batch-size 32 --train-workers 7 --lr 0.1 #/global/cscratch1/sd/kwangmin/dataset/ImageNet/ILSVRC2012



SECONDS=0

# do some work
#srun -u python hybrid_fc_lstm.py --output_dir "3d_output" --epoch 20 --train_batch_size 2
srun -u python hybrid_cnn_lstm.py --epoch 4
duration=$SECONDS
#echo "$(($duration / 60)) minutes and $(($duration % 60)) seconds elapsed."
echo "$duration seconds elapsed."



