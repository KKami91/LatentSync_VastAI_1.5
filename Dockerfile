FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# 기본 패키지 설치
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    ffmpeg \
    libsm6 \
    libxext6 \
    wget \
    curl \
    vim \
    openssh-server \
    && rm -rf /var/lib/apt/lists/*

# 작업 디렉토리 명시적 생성
RUN mkdir -p /workspace

# SSH 서버 설정
RUN mkdir -p /var/run/sshd
RUN echo 'root:password' | chpasswd
RUN sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config

# SSH 로그인 시 작업 디렉토리로 자동 이동
RUN echo 'cd /workspace' >> /root/.bashrc

WORKDIR /workspace
RUN git clone https://github.com/KKami91/LatentSync_VastAI.git && mv LatentSync_VastAI ComfyUI

WORKDIR /workspace/ComfyUI
RUN pip3 install --no-cache-dir -r requirements.txt

WORKDIR /workspace/ComfyUI/custom_nodes/ComfyUI-LatentSyncWrapper
RUN pip3 install --no-cache-dir -r requirements.txt

WORKDIR /workspace/ComfyUI/custom_nodes/ComfyUI-VideoHelperSuite
RUN pip install --no-cache-dir -r requirements.txt

# 필요한 추가 패키지 설치
WORKDIR /workspace/ComfyUI
RUN pip3 install flask gunicorn

RUN pip3 install --upgrade huggingface-hub
WORKDIR /workspace/ComfyUI/custom_nodes/ComfyUI-LatentSyncWrapper
RUN huggingface-cli download ByteDance/LatentSync \
    --local-dir checkpoints --exclude "*.git" "README.md" || echo "Huggingface 모델 다운로드 실패"

COPY LatentSync_api.py /workspace/

RUN mkdir -p /workspace/output /workspace/temp
RUN chmod -R 777 /workspace

EXPOSE 22 8188 5000

# 시작 스크립트 생성
RUN echo '#!/bin/bash\n\
service ssh start\n\
cd /workspace/ComfyUI\n\
python -m main --listen 0.0.0.0 --port 8188 --disable-auto-launch &\n\
cd /workspace\n\
python LatentSync_api.py &\n\
tail -f /dev/null' > /workspace/start.sh

RUN chmod +x /workspace/start.sh

ENTRYPOINT ["/workspace/start.sh"]