#!/bin/bash

echo "\n\n==================Starting TensorRT testing...==================\n\n"

COMPOSE_FILE="docker-compose-triton.yml"

docker compose -f $COMPOSE_FILE run --pull always --rm -v "./:/tmp" --remove-orphans triton_server bash -lc '
cd /tmp

echo "\n\n==================Testing the 512 optimized engine==================\n\n"
/usr/src/tensorrt/bin/trtexec --loadEngine=sources/models/trt/dinov3_vitb16-512.engine --shapes=images:1x3x512x512 --dumpOutput

echo "\n\n==================Testing the 112 optimized engine==================\n\n"
/usr/src/tensorrt/bin/trtexec --loadEngine=sources/models/trt/dinov3_vitb16-112.engine --shapes=images:1x3x224x224 --dumpOutput
'