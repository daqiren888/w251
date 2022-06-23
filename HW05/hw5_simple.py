import torch
import argparse
import torch.optim
import torch.nn as nn
import torch.utils.data
import torchvision.models as models
import torch.backends.cudnn as cudnn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.models as models


epochs=40 
batch_size=128
lr=0.01
gpu=0
num_classes=30

model = models.resnet50(pretrained=True)
model = models.resnet50(num_classes)


torch.cuda.set_device(gpu)
model = model.cuda(gpu)

criterion = nn.CrossEntropyLoss().cuda(gpu)

optimizer = torch.optim.SGD(model.parameters(), lr,
                            momentum=0.9,
                            weight_decay=1e-4)
							
							
train_dataset = datasets.ImageNet('/workspace/imagenet/datasets/imagenet30', 
             split='train',
              transform=transforms.Compose([
              transforms.RandomResizedCrop(224),
              transforms.RandomHorizontalFlip(),
              transforms.ToTensor(),
              transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
           ]))

val_dataset = datasets.ImageNet('/workspace/imagenet/datasets/imagenet30',
              split='val', 
              transform=transforms.Compose([
              transforms.RandomResizedCrop(224),
              transforms.RandomHorizontalFlip(),
              transforms.ToTensor(),
              transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
           ]))


train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size,
        shuffle=True, num_workers=0)

val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size,
        shuffle=False, num_workers=0)
		

def train(epoch):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(gpu), labels.to(gpu)

        optimizer.zero_grad()
        outputs = model(images)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    print('Loss: {:.3}, Acc: {:.4}'.format(train_loss, 100.*correct/total))

def val(epoch):
    global best_acc
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for id, (images, labels) in enumerate(val_loader):
            images, labels = images.to(gpu), labels.to(gpu)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    print('Loss: {:.3} | Acc: {:.4}'.format(val_loss, 100.*correct/total))
			
			
for epoch in range(epochs):
    train(epoch)
    #training is completed 
    val(epoch)

















