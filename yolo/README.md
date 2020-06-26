# YOLO v3, YOLO v3 tiny, YOLO v4

- YOLO v3
    - site: https://pjreddie.com/darknet/yolo/
    - arXiv: https://arxiv.org/abs/1804.02767
    - https://github.com/pjreddie/darknet
- YOLO v4
    - arXiv: https://arxiv.org/abs/2004.10934
    - https://github.com/AlexeyAB/darknet

## create TFlite weight files (dynamic quantization)

- `> ./download_yolo.sh` (it will take too much time)
- `> convert_tflite.py --weights yolov3-tiny.weights --mode dynamic`
- `> convert_tflite.py --weights yolov3.weights --mode dynamic`
- `> convert_tflite.py --weights yolov4.weights --mode dynamic`

## YOLO models cannot be used with TPU

- YOLO v3
    - YOLO v3 uses LeakyReLU as the activate function.
    - LeakyReLU cannot be fully quantized with TensorFlow 2.1
    - TensorFlow nightly (tf\_nightly = TF 2.3?) can fully quantized LeakyReLU
    - But the EdgeTPU compiler does not support TF nightly
    - If the EdgeTPU compiler is updated further, it may be usable YOLO v3 in TPU
- YOLO v4
    - YOLO v4 uses LeakyReLU and Mish as the activate function
    - I think it is difficult to fully quantize Mish
        - because Mish uses exponential

### (cf) create TFlite weight files (full integer quantization)

- `> convert_tflite.py --weights yolov3.weights --mode full`
- `> convert_tflite.py --weights yolov3-tiny.weights --mode full`
- `> convert_tflite.py --weights yolov4.weights --mode full`

### (cf) create TFlite weight files for Edge TPU

- install docker on your PC
- `> build_docker.sh`
- `> compile_edgetpu.sh yolov3.tflite`
- `> compile_edgetpu.sh yolov3_tiny.tflite`
- `> compile_edgetpu.sh yolov4.tflite`
