FROM tensorflow/tensorflow:latest-py3-jupyter

MAINTAINER "Ygor Canalli"

RUN pip3 install pandas sklearn matplotlib optuna aif360 fairlearn pygad mysql-connector-python ray
COPY jupyter_notebook_config.py /root/.jupyter/

RUN apt install wget
RUN wget https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data
RUN wget https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test
RUN wget https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.names
RUN mv adult.data /usr/local/lib/python3.6/dist-packages/aif360/data/raw/adult
RUN mv adult.test /usr/local/lib/python3.6/dist-packages/aif360/data/raw/adult
RUN mv adult.names /usr/local/lib/python3.6/dist-packages/aif360/data/raw/adult

RUN wget https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data
RUN wget https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.doc
RUN mv german.data /usr/local/lib/python3.6/dist-packages/aif360/data/raw/german
RUN mv german.doc /usr/local/lib/python3.6/dist-packages/aif360/data/raw/german

EXPOSE 8888 6006