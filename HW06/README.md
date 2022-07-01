# Homework 6


## Part 1: GStreamer

Q1: What is the difference between a property and a capability?  

- Answer: Properties are used to describe extra information for capabilities. A property consists of a key (a string) and a value. There are different possible value types that can be used. There is a distinct difference between the possible capabilities of a pad , the allowed caps of a pad and lastly negotiatedcaps. we can get values of properties in a set of capabilities by querying individual properties of one structure.

Q2: How are they each expressed in a pipeline?

-Answer: InGstream， pad as the input/output interface of element, the direction is the data flow from the src pad (production data) of one element to the sink pad (consumption data) of another element.


Q3: Explain the following pipeline, that is explain each piece of the pipeline, desribing if it is an element (if so, what type), property, or capability.  What does this pipeline do?

```
gst-launch-1.0 v4l2src device=/dev/video0 ! video/x-raw, framerate=30/1 ! videoconvert ! agingtv scratch-lines=10 ! videoconvert ! xvimagesink sync=false
```

- Answer
gst-launch-1.0 v4l2src 
( Launch Gstream) 
device=/dev/video0 !  video/x-raw, framerate=30/1 ! 
( Catch the pictures from USB webcam; video has 30 frames per sec)  
videoconvert !  
(change color space or format) 
agingtv  scratch-lines=10 ! 
AgingTV ages a video stream in realtime, and adds scratches and dust.
videoconvert ! 
(change color space or format) 
xvimagesink 
（Use XWindow output） 
sync=false
(Synchronisation is disabled entirely by setting the object sync property to FALSE)


- Source code and Gstreamer "server" pipeline used.



















## Part 2: Model optimization and quantization

In lab, you saw to how use leverage TensorRT with TensorFlow.  For this homework, you'll look at another way to levarage TensorRT with Pytorch via the Jetson Inference library (https://github.com/dusty-nv/jetson-inference).

You'll want to train a custom image classification model, using either the fruit example or your own set of classes.

Like in the lab, you'll want to first baseline the your model, looking a the number of images per second it can process.  You may train the model using your Jetson device and the Jetson Inference scripts or train on a GPU eanabled server/virtual machine.  Once you have your baseline, follow the steps/examples outlined in the Jetson Inference to run your model with TensorRT (the defaults used are fine) and determine the number of images per second that are processed.

You may use either the container apporach or build the library from source.

For part 2, you'll need to submit:
- The base model you used
- A description of your data set
- How long you trained your model, how many epochs you specified, and the batch size.
- Native Pytorch baseline
- TensorRT performance numbers

