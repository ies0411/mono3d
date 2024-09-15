
ARG PYTORCH="1.9.0"
ARG CUDA="11.1"
ARG CUDNN="8"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel


ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6+PTX" \
    TORCH_NVCC_FLAGS="-Xfatbin -compress-all" \
    CMAKE_PREFIX_PATH="$(dirname $(which conda))/../" \
    FORCE_CUDA="1"

RUN rm /etc/apt/sources.list.d/cuda.list \
    && rm /etc/apt/sources.list.d/nvidia-ml.list \
    && apt-key del 7fa2af80 \
    && apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub \
    && apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    git zip build-essential cmake python3-dev  python3-pip wget libgl1-mesa-glx libglib2.0-0 &&\
    apt-get clean &&\
    rm -rf /var/lib/apt/lists/*

RUN ln -sv /usr/bin/python3 /usr/bin/python
ENV PATH="/root/.local/bin:${PATH}"
RUN pip install openmim && \
    mim install "mmengine" "mmcv>=2.0.0rc4" "mmdet>=3.0.0"


RUN pip install --user pyyaml scipy opencv-python numba scikit-image tqdm einops flash_attn==0.2.2 fvcore
RUN pip install --upgrade --user numpy
RUN pip install --upgrade pip

# cd lib/models/monodetr/ops/
# bash make.sh


# Install MMDetection3D
# RUN conda clean --all \
#     && git clone https://github.com/open-mmlab/mmdetection3d.git -b dev-1.x /mmdetection3d \
#     && cd /mmdetection3d \
#     && pip install --no-cache-dir -e .

# WORKDIR /mmdetection3d


# RUN conda clean --all \
#     && git clone https://github.com/open-mmlab/mmdetection3d.git -b dev-1.x /mmdetection3d \
#     && cd /mmdetection3d \
#     && pip install --no-cache-dir -e .