#! /bin/bash

echo "Converting all models to ONNX..."

# Installing dependencies
uv sync
uv pip install -r sources/dinov3/requirements.txt


# Convert all models to ONNX
uv run export_model.py --model-type DINOV3 --model-path sources/models/raw/dinov3_vitb16.pth --model-source-code sources/dinov3/ --dino-type dinov3_vitb16 --model-dim 512
uv run export_model.py --model-type DINOV3 --model-path sources/models/raw/dinov3_vitb16.pth --model-source-code sources/dinov3/ --dino-type dinov3_vitb16 --model-dim 112

COMPOSE_FILE="docker-compose-triton.yml"

echo "Starting TensorRT conversion..."

docker compose -f "$COMPOSE_FILE" run --pull always --rm \
  -v "./:/tmp" \
  --remove-orphans \
  triton_server bash -c "
    # 1. Move to the mounted volume so we can find the ONNX files
    cd /tmp && \

    # 2. Create directory for DinoV3-512 and run conversion
    echo 'Converting DinoV3-512...' && \
    /usr/src/tensorrt/bin/trtexec \
      --onnx=sources/models/onnx/dinov3_vitb16-fp32-512.onnx \
      --saveEngine=sources/models/trt/dinov3_vitb16-512.engine \
      --fp16 \
      --minShapes=images:1x3x512x512 \
      --optShapes=images:8x3x512x512 \
      --maxShapes=images:16x3x512x512 \

    # 4. Create directory for DinoV3-112 and run conversion
    echo 'Converting DinoV3-112...' && \
    /usr/src/tensorrt/bin/trtexec \
      --onnx=sources/models/onnx/dinov3_vitb16-fp32-112.onnx \
      --saveEngine=sources/models/trt/dinov3_vitb16-112.engine \
      --fp16 \
      --minShapes=images:1x3x224x224 \
      --optShapes=images:8x3x224x224 \
      --maxShapes=images:16x3x224x224 \
  "

echo "Conversion complete."
