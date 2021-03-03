# YOLO v5

- https://github.com/ultralytics/yolov5

## download PyTorch weights

- `> ./download_weights.sh` (it will take a lot of time)

## download coco dataset for post-training of TFLite (int8 quantization)

- `> ./donwload_coco.sh` (it will also take a lot of time)
    - if you had downloaded the coco dataset when you had compiled YOLO v3, you can use that
    - `> ln -s ../yolo/val2017 .`

## convert weights to TFlite flat buffers

- `> ./convert_tflite.py`

## create TFlite flat buffers for Edge TPU

- install docker on your PC
- run docker
- `> build_docker.sh`
- `> compile_edgetpu.sh yolov5s_int8.tflite`
- `> compile_edgetpu.sh yolov5m_int8.tflite`
- `> compile_edgetpu.sh yolov5l_int8.tflite`
- `> compile_edgetpu.sh yolov5x_int8.tflite`
