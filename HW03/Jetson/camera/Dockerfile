FROM ubuntu:latest
# this is needed to make sure we can see the log output
ENV PYTHONUNBUFFERED=1
WORKDIR /app

# Installing software
#RUN apt update
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y python3 python3-pip

RUN pip3 install paho-mqtt

RUN apt install -y \
    python3-opencv \
    cheese \
    python-pip \
    vim-tiny \
    mosquitto \
    mosquitto-clients \
    libopencv-dev

# Configure to allow remote access
#RUN echo "allow_anonymous true" > /etc/mosquitto/mosquitto.conf
#RUN echo "listener 1883 0.0.0.0" >> /etc/mosquitto/mosquitto.conf
#CMD ["mosquitto","-c","/etc/mosquitto/mosquitto.conf"]

#RUN pip3 install paho-mqtt



#RUN apt-get update && apt-get install -y python3 python3-pip
#RUN a command to intall the MQTT python client 

#RUN apt-get -y update && sudo apt-get -y upgrade
#RUN apt-get install -y python3-numpy
#RUN apt-get install -y python3-opencv
#
#RUN apt-get install -y mosquitto-clients


#RUN pip3 install paho-mqtt

# Copy your listener.py file into the container

# Configure to allow remote access

COPY cameraface.py /app

CMD python3 cameraface.py





