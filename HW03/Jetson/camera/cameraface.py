import paho.mqtt.client as mqtt
import time
import numpy as np
import cv2

LOCAL_MQTT_HOST="mosquitto-service"
LOCAL_MQTT_PORT=1883
LOCAL_MQTT_TOPIC="test_topic"

def on_connect(client, userdata, flags, rc):
    print("connected to local broker with rc: " + str(rc))

local_client = mqtt.Client()
local_client.on_connect = on_connect
local_client.connect(LOCAL_MQTT_HOST, LOCAL_MQTT_PORT, 60)

face_cascade = cv2.CascadeClassifier('/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

print("Detecting Pics")

local_client.loop_start()

time.sleep(1)

while (True):
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        gray = gray[y:y+h, x:x+w]

        img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        img = img[y:y+h, x:x+w]

        try:
            png = cv2.imencode(".png", img)[1]
            msg = bytearray(png)

            local_client.publish(LOCAL_MQTT_TOPIC, payload=msg, qos=0, retain=False)
            print("Transferring Pics")
        except:
            pass

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

time.sleep(1)

local_client.loop_stop()
local_client.disconnect()
cap.release()
cv2.destroyAllWindows()
                            



