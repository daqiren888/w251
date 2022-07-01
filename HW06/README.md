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

(1) gst-launch-1.0 v4l2src \
( Launch Gstream) \
(2) device=/dev/video0 !  video/x-raw, framerate=30/1 ! \
( Catch the pictures from USB webcam; video has 30 frames per sec)  \
(3) videoconvert !  \
(change color space or format) \
(4) agingtv  scratch-lines=10 ! \
AgingTV ages a video stream in realtime, and adds scratches and dust. \
(5) videoconvert ! \
(change color space or format) \ 
(6) xvimagesink \
（Use XWindow output） \
(7) sync=false \
(Synchronisation is disabled entirely by setting the object sync property to FALSE)


Q4: Source code and Gstreamer "server" pipeline used.

-Answer: 

- starts the "server" broadcasting the packets (udp) to the IP Address 127.0.01 on port 8001. The server broadcasts the stream using RTP that hs h265 ecnoded.

gst-launch-1.0 videotestsrc  ! nvvidconv ! omxh265enc insert-vui=1 ! h265parse ! rtph265pay config-interval=1 ! udpsink host=127.0.0.1 port=5000 sync=false -e 

- listens for the packets and decodes the RTP stream and displays it on the screen.

gst-launch-1.0 udpsrc port=5000 ! application/x-rtp, media=video, encoding-name=H265 ! rtph265depay ! h265parse ! nvv4l2decoder ! nvvidconv ! video/x-raw, format=I420 ! nv3dsink -e

 

## Part 2: Model optimization and quantization

- Q1： The base model you used

For the base model, I used native pytorch baseline with resnet18. 

- Q2:A description of your data set

I used dataset with 30 images: I took out 30 images from imagenet dataset, i.e. 30 image-sub-folders from train and val folders repectively.

- Q3:How long you trained your model, how many epochs you specified, and the batch size.

I setup 30 epoches, betch size = 48. The traing run 7.2 hours.

- Q4: Native Pytorch baseline

I record 12 FPS images per second.

- Q5: TensorRT performance numbers

I record 63 FPS, about 5 times faster than the baseline approach. 

