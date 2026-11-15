# MirageBench — eğitim/çıkarım imajı (CUDA + Ultralytics).
# Sentetik üretim için bpy modülü de kurulur; HDRI render'da GPU kullanır.
#
# Kullanım:
#   docker build -t miragebench -f docker/Dockerfile .
#   docker run --gpus all -v $PWD:/workspace -it miragebench bash
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3-pip python3.10-dev \
        libgl1 libglib2.0-0 libxrender1 libxi6 libxkbcommon0 libsm6 \
        git wget && \
    rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.10 /usr/bin/python

WORKDIR /workspace
COPY pyproject.toml requirements.txt ./
COPY miragebench ./miragebench

# Çekirdek + eğitim bağımlılıkları
RUN python -m pip install --upgrade pip && \
    python -m pip install -e ".[train,report]" && \
    python -m pip install bpy

# Çekirdek testlerin import edilebildiğini doğrula (GPU gerektirmez)
RUN python -c "import miragebench; from miragebench.eval.metrics import evaluate; print('miragebench', miragebench.__version__)"

CMD ["bash"]
