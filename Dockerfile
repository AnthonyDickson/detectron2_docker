FROM ubuntu:22.04

RUN apt update && \
    DEBIAN_FRONTEND="noninteractive" apt install -y --no-install-recommends \
    python3 python3-dev python3-pip make cmake ninja-build gcc g++  && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir detectron2 -f \
  https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/torch1.10/index.html
RUN pip install --no-cache-dir torch==1.10.0+cpu torchvision==0.11.1+cpu torchaudio==0.10.0+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html

COPY load_weights.py .
RUN python3 load_weights.py

WORKDIR /app

ENTRYPOINT ["python3"]
