## Model Conversion
The following folder contains scripts that involve converting the raw `.pth` model you got, to a format that is the most performant in a production system. We essentially want to get the fastest inference time, with minimal accuracy loss.

We first need to convert our model from `.pth` to `.onnx`, which is a universal format for machine learning models.
The conversion will result with model_**fp16** and model_**fp32** files, which represent the same model with different datatype formats. <br>
**FP16** Models are essentially very close in accuracy(for inference) and are much lighter and faster to run, and it is recommanded to use. 

Next, we would be converting the `.onnx` file we have to `.engine`, using NVIDIA's TensorRT tool.<br>
TensorRT compiles a model for your specific achitecture(one used at time of compilation), therefore making it very efficient when running on your machine.<br>

## To run
In order to run the script, make sure all your `.pt` models are in the `sources/models/raw` folder. Then, run the following:
```bash
./convert_all_models.sh
```
The script above runs the conversion to `.onnx` and after that to `.engine`.

In order to test the outputs (see if there are NaN values for example), run the following script:
```bash
./test_all_models.sh
```
The script above prints the inference info on the models and in the end prints their outputs

### If you decide to skip some steps

To convert the model from `.pth` to `.onnx` optimized for the shape `{SHAPE}`, run the following command:
```bash
uv run export_model.py \
  --model-type DINOV3 \
  --model-path ORIGINAL.pth \
  --model-source-code sources/dinov3/ \
  --dino-type dinov3_vitb16 \
  --model-dim SHAPE
```
In the example above we run it for DINOv3, therefore the path specified for `--model-source-code` and other arguments are for DINO.
After running the command above, the `.onnx` model will be saved to `sources/models/onnx` with the name `{MODEL_NAME}-fp32-{SHAPE}.onnx`

To convert the model from `.onnx` to `.engine` optimized for the shape `{SHAPE}`, run the following command:
```bash
/usr/src/tensorrt/bin/trtexec \
  --onnx={ONNX_PATH} \
  --saveEngine={ENGINE_PATH} \
  --fp16 \
  --minShapes=images:1x3x512x512 \
  --optShapes=images:8x3x512x512 \
  --maxShapes=images:16x3x512x512 \
```
After running the command above, the `.engine` model will be saved to `sources/models/engine` with the name `{ENGINE_PATH}`

To test performance of TRT model optimized for the shape `{SHAPE}`, run the following command:
```bash
/usr/src/tensorrt/bin/trtexec \
  --loadEngine={ENGINE_PATH} \
  --shapes=images:8x3x{SHAPE}x{SHAPE} \
  --dumpOutput
```

### NOTE!
Before running the `trtexec` commands, make sure you are in the triton container.
You can start the container with [docker-compose-triton.yml](docker-compose-triton.yml) by running:
```bash
docker compose -f docker-compose-triton.yml run --pull always --rm -v "./:/tmp --remove-orphans triton_server bash
```
