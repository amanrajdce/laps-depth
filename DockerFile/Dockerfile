# tensorflow 1.11-gpu, cuda9, cudnn7, ray0.8.4, pytorch-gpu 1.5

FROM floydhub/tensorflow:1.11-gpu.cuda9cudnn7-py3_aws.36

WORKDIR /

RUN pip install ray && \
    pip install numpy==1.16.1 && \
    pip install comet_ml && \
    pip install torch==1.5.0+cu92 torchvision==0.6.0+cu92 -f https://download.pytorch.org/whl/torch_stable.html && \
    pip install requests

EXPOSE 8080
