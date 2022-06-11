FROM ubuntu:latest

# this is needed to make sure we can see the log output
ENV PYTHONUNBUFFERED=1
WORKDIR /app

RUN apt-get update && apt-get install -y python3 python3-pip

# RUN to intall the MQTT python client 

RUN pip3 install paho-mqtt

# Copy .py file into the container

COPY forwarder.py /app

CMD python3 forwarder.py
