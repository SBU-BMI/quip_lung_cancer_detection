FROM ubuntu
MAINTAINER Tahsin Kurc

RUN		apt-get -y update && \
		apt-get install --yes build-essential python3-pip python3-openslide wget zip libgl1-mesa-glx libgl1-mesa-dev && \
		pip install --upgrade pip && \
		curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
	 	sh Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda && \
		eval "$(/Users/jsmith/anaconda/bin/conda shell.YOUR_SHELL_NAME hook)" && \
		conda init && \
		pip install Pillow pymongo && \
		pip install torchvision==0.2.1 && \
		pip install openslide-python && \
		conda install --yes -c conda-forge opencv cudatoolkit=10.1

COPY	. /root/quip_prad_cancer_detection/.

RUN		chmod 0755 /root/quip_prad_cancer_detection/scripts/*

ENV	BASE_DIR="/root/quip_prad_cancer_detection"
ENV	PATH="./":$PATH
WORKDIR	/root/quip_prad_cancer_detection/scripts

CMD ["/bin/bash"]
