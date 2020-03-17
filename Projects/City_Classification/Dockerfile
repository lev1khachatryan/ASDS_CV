FROM python:3.7.3

MAINTAINER Levon Khachatryan <levon.khachatryan.1996.db@gmail.com>

ENV ROOT=/test
RUN mkdir -p $ROOT
ADD . $ROOT
WORKDIR $ROOT

# FROM continuumio/miniconda3:4.6.14

# RUN conda install --yes \
#     nomkl \
#     numpy==1.16.3 \
#     configparser==4.0.2

RUN pip install -r requirements.txt
