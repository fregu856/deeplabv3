# deeplabv3

******
## Paperspace:

To train models and to run pretrained models, you can use an Ubuntu 16.04 P4000 VM with 250 GB SSD on Paperspace. Below I have listed what I needed to do in order to get started, and some things I found useful.

- Install docker-ce:
- - $ curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
- - $ sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"
- - $ sudo apt-get update
- - $ sudo apt-get install -y docker-ce

- Install CUDA drivers:
- - $ CUDA_REPO_PKG=cuda-repo-ubuntu1604_9.1.85-1_amd64.deb
- - $ wget -O /tmp/${CUDA_REPO_PKG} http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/${CUDA_REPO_PKG}
- - $ sudo dpkg -i /tmp/${CUDA_REPO_PKG}
- - $ sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub
- - $ rm -f /tmp/${CUDA_REPO_PKG}
- - $ sudo apt-get update
- - $ sudo apt-get install cuda-drivers
- - Reboot the VM.

- Install nvidia-docker:
- - $ wget -P /tmp https://github.com/NVIDIA/nvidia-docker/releases/download/v1.0.1/nvidia-docker_1.0.1-1_amd64.deb
- - $ sudo dpkg -i /tmp/nvidia-docker*.deb && rm /tmp/nvidia-docker*.deb
- - $ sudo nvidia-docker run --rm nvidia/cuda nvidia-smi

- Download the PyTorch 0.4 docker image:
- - $ sudo docker pull pytorch/pytorch:0.4_cuda9_cudnn7

- Create start_docker_image.sh containing:
```
#!/bin/bash

# DEFAULT VALUES
GPUIDS="0"
NAME="paperspace_GPU"


NV_GPU="$GPUIDS" nvidia-docker run -it --rm \
        -p 5584:5584 \
        --name "$NAME""$GPUIDS" \
        -v /home/paperspace:/root/ \
        pytorch/pytorch:0.4_cuda9_cudnn7 bash
```

- Inside the image, /root/ will now be mapped to /home/paperspace (i.e., $ cd -- takes you to the regular home folder). 

- To start the image:
- - $ sudo sh start_docker_image.sh 
- To commit changes to the image:
- - Open a new terminal window.
- - $ sudo docker commit paperspace_GPU0 pytorch/pytorch:0.4_cuda9_cudnn7
- To stop the image when it’s running:
- - $ sudo docker stop paperspace_GPU0
- To exit the image without killing running code:
- - Ctrl + P + Q
- To get back into a running image:
- - $ sudo docker attach paperspace_GPU0
- To open more than one terminal window at the same time:
- - $ sudo docker exec -it paperspace_GPU0 bash

- To install the needed software inside the docker image:
- - $ apt-get update
- - $ apt-get install nano
- - $ apt-get install sudo
- - $ apt-get install wget
- - $ sudo apt install unzip
- - $ sudo apt-get install libopencv-dev
- - $ pip install opencv-python
- - $ python -mpip install matplotlib
- - Commit changes to the image (otherwise, the installed packages will be removed at exit!)

- Do the following outside of the docker image:
- - $ --

- - $ git clone https://github.com/fregu856/deeplabv3.git

- - Download the Cityscapes dataset:
- - - Register on the [website](https://www.cityscapes-dataset.com/).
- - - $ wget --keep-session-cookies --save-cookies=cookies.txt --post-data 'username=XXXXX&password=YYYYY&submit=Login' https://www.cityscapes-dataset.com/login/ *(where you replace XXXXX with your username, and YYYYY with your password)*
- - - $ wget --load-cookies cookies.txt --content-disposition https://www.cityscapes-dataset.com/file-handling/?packageID=1
- - - $ wget --load-cookies cookies.txt --content-disposition https://www.cityscapes-dataset.com/file-handling/?packageID=3

- - - $ unzip gtFine_trainvaltest.zip
- - - $ unzip leftImg8bit_trainvaltest.zip

- - - $ mkdir deeplabv3/data
- - - $ mkdir deeplabv3/data/cityscapes
- - - $ mv gtFine deeplabv3/data/cityscapes
- - - $ mv leftImg8bit deeplabv3/data/cityscapes

****
****

****

### TODO:

- SSH into the paperspace server.
- $ sudo sh start_docker_image.sh
- $ cd --
- $ python deeplabv3/preprocess_data.py

### TODO Train:

- SSH into the paperspace server.
- $ sudo sh start_docker_image.sh
- $ cd --
- $ python deeplabv3/preprocess_data.py *(Only need to this once!)*
- $ python deeplabv3/train.py
