# YOLO v3, YOLO v3 tiny, YOLO v4

- YOLO v3
    - site: https://pjreddie.com/darknet/yolo/
    - arXiv: https://arxiv.org/abs/1804.02767
    - https://github.com/pjreddie/darknet
- YOLO v4
    - arXiv: https://arxiv.org/abs/2004.10934
    - https://github.com/AlexeyAB/darknet

## download darknet weights

- `> ./download_weights.sh` (it will take a lot of time)

## download coco dataset for post-training of TFLite (int8 quantization)

- `> ./donwload_coco.sh` (it will also take a lot of time)

## convert darknet weights to TFlite flat buffers

- `> ./convert_tflite.py`

## create TFlite flat buffers for Edge TPU

- install docker on your PC
- run docker
- `> build_docker.sh`
- `> compile_edgetpu.sh yolov3-tiny_int8.tflite`
- `> compile_edgetpu.sh yolov3_int8.tflite`
- YOLO V4 cannot compile for EdgeTPU due to `mish`
    - once this issue is resolved, you should be able to compile with the following command
    - `> compile_edgetpu.sh yolov4_int8.tflite`
    - but I think it is unpossible because `mish` uses exp and log
