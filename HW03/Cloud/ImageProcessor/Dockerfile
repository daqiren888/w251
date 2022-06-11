FROM ubuntu:latest

ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive


WORKDIR /app
RUN apt-get update && apt-get install -y python3 python3-pip

#RUN to intall the MQTT python client 

RUN pip3 install paho-mqtt

RUN pip install boto3

# Installing software

RUN  apt-get install -y python3-numpy

RUN apt install -y \
    python3-opencv \
    cheese \
    python-pip \
    vim-tiny \
    mosquitto \
    mosquitto-clients \
    libopencv-dev

# Copy .py file into the container

COPY facesubs.py /app
CMD python3 facesubs.py
