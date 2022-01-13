FROM tensorflow/tensorflow:latest-gpu-py3-jupyter

MAINTAINER "Ygor Canalli"

RUN pip3 install pandas sklearn matplotlib optuna aif360
RUN ["mkdir", "notebooks"]
COPY aif360/jupyter_notebook_config.py /root/.jupyter/

EXPOSE 8888 6006

VOLUME /notebooks

