# age and gender estimator

Age and Gender Estimation with WideResNet

github: https://github.com/yu4u/age-gender-estimation

## setup

- download the pretrained weights
    - `> download_pretrained.sh`
- create full quantized TFlite weights
    - `> convert_tflite.py --mode full`
- install docker and launch docker
- build docker image for compile_edgetpu
    - `> build_docker.sh`
- compile TFlite weights
    - `> compile_edgetpu.sh agender.tflite`
- create dynamic quantized TFlite weights (for non-TPU)
    - `> convert_tflite.py --mode dynamic`

## how to use

- try `face_agender.py`
