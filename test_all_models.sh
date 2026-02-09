#!/bin/bash

echo "Starting TensorRT testing..."

COMPOSE_FILE="docker-compose-triton.yml"

docker compose -f $COMPOSE_FILE run --pull always --rm -v "./:/tmp" --remove-orphans triton_server bash -lc '
cd /tmp
/usr/src/tensorrt/bin/trtexec --loadEngine=sources/models/trt/dinov3_vitb16-512.engine --shapes=images:1x3x512x512 --dumpOutput
/usr/src/tensorrt/bin/trtexec --loadEngine=sources/models/trt/dinov3_vitb16-112.engine --shapes=images:1x3x224x224 --dumpOutput
'