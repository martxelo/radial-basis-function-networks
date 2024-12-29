FROM python:3.12-slim

RUN apt update && apt install -y build-essential && pip install --upgrade pip && apt clean

RUN mkdir /opt/project
WORKDIR /opt/project
COPY . /opt/project

RUN pip install -r requirements.txt