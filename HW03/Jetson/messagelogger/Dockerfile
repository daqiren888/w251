FROM ubuntu:latest

ENV PYTHONUNBUFFERED=1

WORKDIR /app
RUN apt-get update && apt-get install -y python3 python3-pip

#RUN a command to intall the MQTT python client 

RUN pip3 install paho-mqtt

COPY messagelogger.py /app

CMD python3 messagelogger.py
