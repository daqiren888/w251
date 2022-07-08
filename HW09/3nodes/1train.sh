python main_amp.py  -a resnet18 --dist-url 'tcp://13.56.236.200:30000' --dist-backend 'nccl' --multiprocessing-distributed --world-size 3 --rank 0   ../../datasets/imagenet 
