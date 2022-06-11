import paho.mqtt.client as mqtt
import time
import logging
import cv2
import uuid
import numpy as np
import os
import os.path
import boto3
from botocore.exceptions import ClientError

# In this work, I use 2 methods to upload the image file to s3fs:
#
# (1) use boto3 to transfer the image file to s3fs.
#
# (2) mount my s3fs folder as a local shared folder, namely /mys3bucket,
# and save the image file into the shared folder /mys3bucket dicrectly.
#

# (Method #2) The path to my local shared folder mounted to s3fs:
save_path = '/mys3bucket/'

# (Method #1) function to transfer to s3fs: 
s3 = boto3.client("s3", aws_access_key_id='AKIAYYJTXWM2J2HMQHNI', aws_secret_access_key='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')



LOCAL_MQTT_HOST="mosquitto-service"
LOCAL_MQTT_PORT=1883
LOCAL_MQTT_TOPIC="cloud_face"

bucket_name = 'daqihw3'

def on_connect(client, userdata, flags, rc):
    print("connected to receiver with rc: " + str(rc))
    client.subscribe(LOCAL_MQTT_TOPIC)

picnumber = 0


def on_message(client, userdata, msg):
    print("message received. Decoding...")
    # detecting images
    
    global picnumber
    decode = np.frombuffer(msg.payload, dtype=np.uint8)
    picture = cv2.imdecode(decode, flags=1)
    picnumber += 1   
    
    # Edit picture file name 

    imagename = "face"+ str(picnumber)
    
    #------------------------------------
    # Method #1: transfer image file 

    completeName = os.path.join(imagename+'.jpg')
    # Write image 
    cv2.imwrite(completeName, picture)
    #uploca image
    s3.upload_file( Filename=completeName, Bucket="drenhw3", Key= os.path.basename(completeName))
    #------------------------------------
    


    #------------------------------------
    # Mthod #2 save file to shared folder mounted to s3fs
 
    #completeName = os.path.join(save_path, filename+'.jpg')
    ## Write image to shared folder, it iwll appear in s3fs bucket.
    # cv2.imwrite(completeName, picture)
    #-------------------------------------


local_client = mqtt.Client()
local_client.on_connect = on_connect
local_client.on_message = on_message
local_client.connect(LOCAL_MQTT_HOST, LOCAL_MQTT_PORT, 60)

local_client.loop_forever()


