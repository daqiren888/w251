#python main_amp.py  -a resnet18 --dist-url 'tcp://13.56.236.200:30000' --dist-backend 'nccl' --world-size 1 --rank 0 --gpu=0  ../../datasets/imagenet 

python main_amp.py -a resnet18  --dist-url 'tcp://13.56.236.200:30000' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 ../../datasets/imagenet
