python main_amp.py -a resnet18  --dist-url 'tcp://52.53.127.120:30000' --dist-backend 'nccl' --multiprocessing-distributed --world-size 2 --rank 0  ../../datasets/imagenet 

