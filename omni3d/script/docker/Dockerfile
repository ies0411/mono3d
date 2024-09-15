FROM nvidia/cuda:11.1.1-devel-ubuntu20.04

ENV TORCH_CUDA_ARCH_LIST="3.5;5.0;6.0;6.1;7.0;7.5;8.0;8.6+PTX;Kepler;Kepler+Tesla;Maxwell;Maxwell+Tegra;Pascal;Volta;Turing;8.6" \
    TORCH_NVCC_FLAGS="-Xfatbin -compress-all" \
    CMAKE_PREFIX_PATH="$(dirname $(which conda))/../" \
    FORCE_CUDA="1"

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
  git zip build-essential libssl-dev python3-dev python3-pip \
  locales curl gnupg2 lsb-release \
  cmake ninja-build wget ca-certificates ffmpeg libsm6 libxext6 x11-apps libgl1-mesa-dev \
  libgl1-mesa-glx libglib2.0-0 \
  xorg xrdp xserver-xorg mesa-utils xauth gdm3 bzip2\
  && rm -rf /var/lib/apt/lists/*

RUN ln -sv /usr/bin/python3 /usr/bin/python
ENV PATH="/root/.local/bin:${PATH}"

RUN pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install --user cython opencv-python  pillow==9.5 scipy pandas \
    && pip install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI' \
    && python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.8/index.html \
    # && pip install --user "git+https://github.com/facebookresearch/pytorch3d.git@stable"
    && pip install --user "git+https://github.com/facebookresearch/pytorch3d.git@v0.6"
