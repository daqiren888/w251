
# Homework 5 - Deep Learning Frameworks 

### Platform 

I setup a T4 GPU and 1 TB of space the Nvidia Deep Learning AMI, and use the latest nvidia pytorch container.  The ImageNet dataset is downloaded into my VM.  Prepare the dataset by create train and val subdirectories and move the train and val tar files to their respective locations.  Under the train and val folders, there is one directory for each class and that the samples for that class are under that directory.

I made a hw5.sh file to leave the training process run at the backend using commandline: 

setsid sh hw5.sh >> hw5_train.log  

### Coding

I used PyTorch in this work with resnet18 (hw5.py) and resnet50 (hw5_simple.py) , referred the lab practice of week05 in the coding work. I adapt the code that we have discussed in the labs to the training of imagenet. The number of classes and image sizes and the image transforms  have been ajusted according the imagenet. 

### Key decisions to consider

- Which architecture to choose? 
Resnet18 was used in my first code HW05.py
Resnet50 was used in my simple version HW05-simply.py 

- Which optimizer to use? 
I used Stochastic gradient descent (SGD)

- What should the learning rate be?  
I tried different start LR from 0.001 - 0.0001. Then I used cosine scheduler. 

### Traning

I setup Epoch = 30, the top 1 accuracy reached 61% when epoch = 27. 


### Extra credit challange
I tried using resnet50 (hw5_simple.py) with minor modificaitons. 

### Submission

Training logs: hw5_train.log

Coding and other files are on github
 
https://github.com/daqiren888/w251/tree/main/HW05
