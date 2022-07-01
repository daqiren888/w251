# %%
import os
import random
import shutil
import time
import warnings

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim

import torch.utils.data
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models



# %%
SEED=1

# %%
random.seed(SEED)
torch.manual_seed(SEED)
cudnn.deterministic = True

# %%
START_EPOCH = 0

# %%
"""
### Set the architecture to resnet 18 below
"""

# %%
##########################
ARCH = "resnet18" # set the architecture to RESNET 18
# please look up how to do that
########################
EPOCHS = 30 
LR = 0.1
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4
PRINT_FREQ = 50
TRAIN_BATCH=48
VAL_BATCH=48
WORKERS=0
# TRAINDIR="/workspace/data/imagenet2012/train"
# VALDIR="/workspace/data/imagenet2012/val"

# %%
# TRAINDIR="/CINIC/train"
# VALDIR="/CINIC/valid"

# %%
"""
### Check if cuda is available here
"""

# %%
# check if cuda is available in this cell
# if it is not available, you should not go forward!
torch.cuda.is_available()

# %%
"""
### Assign your GPU below
"""

# %%
# Assign your GPU in this cell
GPU = torch.cuda.current_device()

# %%
# set your active device to your GPU in this cell
device = torch.device(GPU)
print('device =',device)

# %%
# enable algorithm optimization
cudnn.benchmark = True

# %%
"""
### Fill in the heart of the train section below
"""

# %%
def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    ######################
    # switch model to train mode here
    model.train()
    ################

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        #####################
        # send the images to cuda device
        images = images.cuda(GPU, non_blocking=True)
        # send the target to cuda device
        target = target.cuda(GPU, non_blocking=True)

        # compute output
        output = model(images)


        # compute loss 
        loss = criterion(output, target)


        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        
        #### zero out gradients in the optimier
        optimizer.zero_grad()
        
        ## backprop!
        loss.backward()
        
        # update the weights!
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % PRINT_FREQ == 0:
            progress.display(i)

# %%
"""
#### Fill in the validate section below
"""

# %%
def validate(val_loader, model, criterion):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            
            
            ### send the images and target to cuda
            # send the images to cuda device
            images = images.cuda(GPU, non_blocking=True)
            # send the target to cuda device
            target = target.cuda(GPU, non_blocking=True)

            # compute output
            output = model(images)

            # compute loss
            loss = criterion(output, target)


            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % PRINT_FREQ == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg

# %%
"""
### Save the checkpoint
"""

# %%
def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    # save the model state!
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

# %%
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

# %%
class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

# %%
# if we are adjusting the LR manually use this
def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = LR * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# %%
def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

# %%
imagenet_mean_RGB = [0.47889522, 0.47227842, 0.43047404]
imagenet_std_RGB = [0.229, 0.224, 0.225]
cinic_mean_RGB = [0.47889522, 0.47227842, 0.43047404]
cinic_std_RGB = [0.24205776, 0.23828046, 0.25874835]
cifar_mean_RGB = [0.4914, 0.4822, 0.4465]
cifar_std_RGB = [0.2023, 0.1994, 0.2010]

# %%
#normalize = transforms.Normalize(mean=imagenet_mean_RGB, std=imagenet_std_RGB)

normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
)

# %%
IMG_SIZE = 256
# IMG_SIZE = 224
NUM_SIZE = 1000 

# %%
"""
### Initialize the model using the architecture you selected above
"""

# %%
# select the model
model = models.__dict__[ARCH]()
model.fc = nn.Linear(model.fc.in_features, NUM_SIZE)

# %%
"""
### Send the model to the cuda device
"""

# %%
# send the model to the cuda device.. 
# model = model.to(device)
model.cuda(GPU)

# %%
"""
### Instantiate the loss to cross entropy
"""

# %%
# use the cross-entropy loss
criterion = nn.CrossEntropyLoss().cuda(GPU)

# %%
"""
### Instantiate the optimizer to SGD
"""

# %%
# use SGD .. use the momentum and weight decay vars
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=LR,
    momentum=MOMENTUM,
    weight_decay=WEIGHT_DECAY
)

# %%
"""
#### Create the learning rate scheduler
"""

# %%
# use CosineAnnealingLR
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS)

# %%
#transform_train = transforms.Compose([
#    transforms.RandomCrop(256, padding= None),
#    transforms.RandomHorizontalFlip(),
#    transforms.ToTensor(),
#    transforms.Normalize(cifar_mean_RGB, cifar_std_RGB),
#])

# %%
"""
### Create the train dataset object
"""

# %%
# use torchvision.datasets.CIFAR10



transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
])



#rain_dataset = torchvision.datasets.ImageNet('/workspace/imagenet/datasets/imagenet')
#train_dataset = torchvision.datasets.ImageNet(".", train=True, download=True, transform=transform_train)
train_dataset = torchvision.datasets.ImageNet('/jetson_inference/HW06/imagenet10', split='train', transform=transform_train)

# %%
"""
### Create the val dataset object
"""

# %%
# use torchvision.datasets.CIFAR10
#transform_val = transforms.Compose([
#    transforms.ToTensor(),
#    transforms.Normalize(cifar_mean_RGB, cifar_std_RGB),
#])


transform_val = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])



#al_dataset = torchvision.datasets.ImageNet('/workspace/imagenet/datasets/imagenet')
#val_dataset = torchvision.datasets.CIFAR10(".", train=False, download=True, transform=transform_val)
val_dataset = torchvision.datasets.ImageNet('/jetson_inference/HW06/imagenet10', split='val', transform=transform_val)

# %%
"""
### Create the train dataloader
"""

# %%
# fill this in
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=TRAIN_BATCH, shuffle=True, pin_memory=True, num_workers=WORKERS)
#data_loader = torch.utils.data.DataLoader(imagenet_data,
#                                          batch_size=4,
#                                          shuffle=True,
#                                          num_workers=args.nThreads)
# %%
"""
### Create the c
"""

# %%
# fill this in..
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=VAL_BATCH, shuffle=False, pin_memory=True, num_workers=WORKERS) 

# %%
best_acc1 = 0

# %%
for epoch in range(START_EPOCH, EPOCHS):
#    adjust_learning_rate(optimizer, epoch)

    # train for one epoch
    train(train_loader, model, criterion, optimizer, epoch)

    # evaluate on validation set
    acc1 = validate(val_loader, model, criterion)

    # remember best acc@1 and save checkpoint
    is_best = acc1 > best_acc1
    best_acc1 = max(acc1, best_acc1)


    save_checkpoint({
        'epoch': epoch + 1,
        'arch': ARCH,
        'state_dict': model.state_dict(),
        'best_acc1': best_acc1,
        'optimizer' : optimizer.state_dict(),
    }, is_best)
    
    scheduler.step()
    print('lr: ' + str(scheduler.get_last_lr()))

# %%


# %%
