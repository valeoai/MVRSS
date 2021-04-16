FROM ubuntu:18.04

RUN apt-get update && apt-get install -y wget bzip2 python3-pip
RUN wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
RUN bash miniconda.sh -b -p /opt/conda && rm miniconda.sh
ENV PATH="/opt/conda/bin:${PATH}"
RUN conda config --set always_yes yes
RUN conda install python=3.6

RUN pip install numpy==1.17.4 Pillow>=8.1.1 scikit-image==0.16.2 scikit-learn==0.22 scipy==1.3.3 tensorboard==2.0.2 torch==1.3.1 torchvision==0.4.2

COPY ./ ./MVRSS
RUN pip install -e ./MVRSS

ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=all

WORKDIR ./MVRSS

# Download the CARRADA Dataset
# RUN mkdir /home/datasets_local
# RUN wget -P /home/datasets_local http://download.tsi.telecom-paristech.fr/Carrada/Carrada.tar.gz
# RUN tar -xvzf /home/datasets_local/Carrada.tar.gz -C /home/datasets_local
