# YOLO v3, YOLO v3 tiny, YOLO v4

- YOLO v3
    - arXiv: https://arxiv.org/abs/1804.02767
    - https://github.com/pjreddie/darknet
- YOLO v4
    - arXiv: https://arxiv.org/abs/2004.10934
    - https://github.com/AlexeyAB/darknet
- The github repository I mostly reference
    - https://github.com/hunglc007/tensorflow-yolov4-tflite

Python codes in this folder is almost the same
with the above hunglc007's repository.
But the changing points are below.

- set image size as 320
- input_details and output_details are the same with [Google's SSD](https://github.com/google-coral/examples-camera)

## create TFlite weights files (full integer quantization)

- `> ./download_yolo.sh` (it will take too much time)
- `> ./download_coco.sh` (it will take too much time)
- `> python convert_tflite.py --weights yolov3.weights`
- `> python convert_tflite.py --weights yolov3-tiny.weights`
- `> python convert_tflite.py --weights yolov4.weights`

## create TFlite weights files for Edge TPU

- install docker on your PC
- `> build_docker.sh`
- `> compile_edgetpu.sh yolov3.tflite`
- `> compile_edgetpu.sh yolov3_tiny.tflite`
- `> compile_edgetpu.sh yolov4.tflite`
