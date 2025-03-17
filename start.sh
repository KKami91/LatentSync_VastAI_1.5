#!/bin/bash

# SSH 서버 시작 (원격 접속을 위해 필요)
service ssh start

# LatentSync API 서버 시작 (이것이 핵심 서비스)
cd /workspace/ComfyUI
python3 LatentSync_api.py &

# 컨테이너 유지
tail -f /dev/null